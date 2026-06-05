from __future__ import annotations

from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
import torch
import numpy as np
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms, quat_mul, quat_from_euler_xyz

# Non-privledged critic. Vanilla PPO
@configclass
class StackShakeEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20 # Will need to adjust
    decimation = 2
    action_space = 7 # (POS, ROT, Gripper)
    observation_space = 65 # finger positions, end effector pose, flange pos, gripper width, IK pose, cube poses/vel, relative vectors (ee->cube, ee->cube, cube->cube)
    state_space = 0 # symmetric

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/MyCobot280",
        spawn=sim_utils.UsdFileCfg(
            usd_path=r"C:\Users\wesle\custom_assets\usd_assets\mycobot_280_m5_adaptive_gripper.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, 
                enabled_self_collisions=True, 
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=1 # this is how many times we compute the position and velcoity
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0,
                "joint2": 0,
                "joint3": 0,
                "joint4": 0,
                "joint5": 0,
                "joint6": 0,
                "gripper_base_to_gripper_left3": 0.0, # singular drive joint
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "arm_actuators": ImplicitActuatorCfg(
                joint_names_expr=[
                    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
                ],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.79253, # 160 deg per second into rad per sec
                stiffness=500.0,
                damping=60.0,
            ),
            "gripper_actuators": ImplicitActuatorCfg(
                joint_names_expr=[
                    "gripper_base_to_gripper_left3"
                ],
                effort_limit_sim=100.0,
                stiffness=1e6, # 2e3 currently results in around 0.77 - 0.89 range for the angle in radians
                damping=100,
            ),
        },
    )

    # cube one
    cube_one = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeOne",
        spawn=sim_utils.CuboidCfg(
            size=(0.0225, 0.0225, 0.0225),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.125),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.1,0.1,1.0)),
        ),
    )

    # cube two
    cube_two = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeTwo",
        spawn=sim_utils.CuboidCfg(
            size=(0.0225, 0.0225, 0.0225),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.125),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.1,0.1,1.0)),
        ),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 1.0
    dof_velocity_scale = 0.1

    # hardware constants
    GRIPPER_WIDTH_TO_ANGLE_CUBIC_COEF = [15784.784389302775, -1051.1819479393648, 41.5177329581917, -1.0013858239467686]
    GRIPPER_MAX_GRASP = 0.045 # m
    GRIPPER_MIN_GRASP = 0.01 # m 
    GRIPPER_SMOOTHING = 0.05
    GRIPPER_MAX_GRASP = 0.045 # m
    GRIPPER_MIN_GRASP = 0.01 # m 
    GRIPPER_SMOOTHING = 0.02

    GRIPPER_NAMES = [
        "gripper_base_to_gripper_left3", # IMORTANT: put driver joint first
        "gripper_base_to_gripper_left2",
        "gripper_left3_to_gripper_left1",
        "gripper_base_to_gripper_right3",
        "gripper_base_to_gripper_right2",
        "gripper_right3_to_gripper_right1",
    ]

    # bounds
    x_bound = 0.5
    y_bound = 0.5
    z_bound = 1.0

    # reward scales
    reach_reward_weight = 1.5

    lift_reward_weight = 30.0

    target_reward_weight = 10.0
    target_fine_reward_weight = 20.0
    target_close_reward_weight = 10.0
    target_close_bonus_weight = 200.0

    stack_coarse_reward_weight = 8.0
    stack_fine_reward_weight = 12.0

    xy_alignment_reward_weight = 10.0
    height_alignment_reward_weight = 6.0

    release_reward_weight = 15.0

    success_reward_weight = 50.0

    action_penalty_final_rate = 1e-2
    joint_vel_penalty_final_rate = 1e-2

    # observation scales for nromalization
    pos_obs_scale = 0.5
    quat_obs_scale = 1.0
    vel_obs_scale = 2.0
    ang_vel_obs_scale = 3.0
    gripper_width_scale = GRIPPER_MAX_GRASP
    joint_vel_scale = 2.79253
    action_obs_scale = 1.0
    gripper_action_obs_scale = 10

class StackShakeEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    
    cfg: StackShakeEnvCfg

    def __init__(self, cfg: StackShakeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # this slows down the finger joints
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        # Look up the actual joint indices instead of body indices
        left_joint_idx, _ = self._robot.find_joints("gripper_base_to_gripper_left3")
        right_joint_idx, _ = self._robot.find_joints("gripper_base_to_gripper_right3") # Double-check your right joint name layout

        # Extract the integer index out of the returned list and apply the scale
        self.robot_dof_speed_scales[left_joint_idx[0]] = 0.1
        self.robot_dof_speed_scales[right_joint_idx[0]] = 0.1
        
        # set the targets to zero
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # joint idxs
        self.arm_joint_ids = torch.arange(6, device=self.device) # they are first six joints
        gripper_joint_ids = []
        for name in self.cfg.GRIPPER_NAMES:
            ids, _ = self._robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(f"Joint {name} not found uniquely")
            gripper_joint_ids.append(ids[0])

        self.gripper_joint_ids = torch.tensor(gripper_joint_ids, device=self.device, dtype=torch.long)
        self.gripper_width = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.gripper_q = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        stage = get_current_stage()

        # set up the positions of parts
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/MyCobot280/MyCobot280/gripper_base")),
            self.device
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/MyCobot280/MyCobot280/pad_left")),
            self.device
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/MyCobot280/MyCobot280/pad_right")),
            self.device
        )


        # -- Setting up all positions at init
        # in between the fingers
        end_effector_pose = torch.zeros(7, device=self.device) # pos + quat
        end_effector_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2
        end_effector_pose[3:7] = hand_pose[3:7]

        self.hand_link_idx = self._robot.find_bodies("gripper_base")[0][0] # may need to revert to joint 6 center?
        self.left_finger_link_idx = self._robot.find_bodies("pad_left")[0][0] 
        self.right_finger_link_idx = self._robot.find_bodies("pad_right")[0][0]

        # the flange/hand of the robot (indepdent of the adaptive fingers)
        self.robot_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # in between the finger pads (TCP)
        self.robot_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_ee_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # for finger pad obs
        self.left_pad_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_pad_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # gripper filtered q
        self._filtered_gripper_q = torch.zeros(
            (self.num_envs,),
            device=self.device
        )
        # setup our inverse kinematic controller
        self.ik_controller_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls"
        )
        self.ik_controller = DifferentialIKController(cfg=self.ik_controller_cfg, num_envs=self.num_envs,device=self.device)
        self.ik_commands = torch.zeros(self.scene.num_envs, self.ik_controller.action_dim, device=self.device)
        self.ik_commands[:] = 0.0 # this is the starting command pose, so replace with default pos eventually I think

        # cube setup
        self.cube_one_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_two_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_one_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.cube_two_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.cube_one_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_one_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_two_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_two_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        self.ee_to_cube_one = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_to_cube_two = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_one_to_cube_two = torch.zeros((self.num_envs, 3), device=self.device)
        
        # get_rewards() stored data
        self.count = 0
        self.current_action = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.past_action = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)

        # used for rolling reset time
        self.success_window = 50 # tune this
        self.base_episode_length = int(self.cfg.episode_length_s / self.cfg.sim.dt / self.cfg.decimation)

        # rolling history per env
        self.success_history = torch.zeros(
            (self.num_envs, self.success_window), device=self.device, dtype=torch.float32
        )
        self.success_write_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # episode level success flag
        self.episode_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # per-env trunction length chosen at reset time
        self.adaptive_truncation_steps = torch.full(
            (self.num_envs,), self.base_episode_length, device=self.device, dtype=torch.long
        )
        # multiplier to adjust convergance end time (magic number)
        self.max_horizon_multiplier = 180.0


    # pull in the objects in and setup env
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)

        self.arm_ik_actions = 3 + 3 # (X, Y, Z), (Yaw, Pitch, Roll)
        self.num_arm_joints = 6 # DOF arm
        self.num_gripper_joints = 1 # one driver joint
        self.num_actions = self.arm_ik_actions + self.num_gripper_joints # gripper + 6 DOF

        self._cube_one = RigidObject(self.cfg.cube_one)
        self._cube_two = RigidObject(self.cfg.cube_two)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collision for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add after clone (may be wrong...)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["cube_one"] = self._cube_one
        self.scene.rigid_objects["cube_two"] = self._cube_two

        # append objects to randomization list
        self.randomized_objects = [self._cube_one, self._cube_two]
        
        # get pushed randomly
        self.pushed_objects = [self._cube_one, self._cube_two]
    
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        # clamps actions, scales them, clamps them again, applies them to the dof targets
        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        # [dx, dy, dz, droll, dpitch, dyaw, gripper]
        dpos = self.actions[:, 0:3] * 0.01
        drot = self.actions[:, 3:6] * 0.05

        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self.hand_link_idx - 1, :, self.arm_joint_ids
        ]

        hand_pose_w = self._robot.data.body_pose_w[:, self.hand_link_idx]
        root_pose_w = self._robot.data.root_pose_w
        joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]

        # EE pose in robot root frame
        hand_pos_b, hand_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            hand_pose_w[:, 0:3],
            hand_pose_w[:, 3:7],
        )

        if not hasattr(self, "_ik_initialized"):
            self.ik_commands[:, 0:3] = hand_pos_b
            self.ik_commands[:, 3:7] = hand_quat_b
            self._ik_initialized = True

        # delta position control
        self.ik_commands[:, 0:3] += dpos

        # clamp workspace
        self.ik_commands[:, 0] = torch.clamp(self.ik_commands[:, 0], -0.35, 0.35)
        self.ik_commands[:, 1] = torch.clamp(self.ik_commands[:, 1], -0.35, 0.35)
        self.ik_commands[:, 2] = torch.clamp(self.ik_commands[:, 2], 0.02, 0.5)

        delta_quat = quat_from_euler_xyz(
            drot[:, 0],  # roll
            drot[:, 1],  # pitch
            drot[:, 2],  # yaw
        )

        # Current target quaternion
        current_quat = self.ik_commands[:, 3:7]

        # Apply local rotation update
        new_quat = quat_mul(current_quat, delta_quat)

        # Normalize
        new_quat = new_quat / torch.norm(new_quat, dim=-1, keepdim=True)

        # Store back
        self.ik_commands[:, 3:7] = new_quat

        # send target to IK controller
        self.ik_controller.set_command(self.ik_commands)

        # solve IK
        joint_pos_des = self.ik_controller.compute(
            hand_pos_b,
            hand_quat_b,
            jacobian,
            joint_pos,
        )

        self.robot_dof_targets[:, self.arm_joint_ids] = torch.clamp(
            joint_pos_des,
            self.robot_dof_lower_limits[self.arm_joint_ids],
            self.robot_dof_upper_limits[self.arm_joint_ids],
        )

        # gripper stuff
        gripper_actions = self.actions[:, 6]
        target_width = (gripper_actions + 1.0) / 2.0 * (self.cfg.GRIPPER_MAX_GRASP - self.cfg.GRIPPER_MIN_GRASP) + self.cfg.GRIPPER_MIN_GRASP

        w = torch.clamp(target_width, min=self.cfg.GRIPPER_MIN_GRASP)
        q = self.cfg.GRIPPER_WIDTH_TO_ANGLE_CUBIC_COEF[0]
        for c in self.cfg.GRIPPER_WIDTH_TO_ANGLE_CUBIC_COEF[1:]:
            q = q * w + c
        
        q_clamped = torch.clamp(
            q, 
            self.robot_dof_lower_limits[self.gripper_joint_ids[0]], 
            self.robot_dof_upper_limits[self.gripper_joint_ids[0]]
        )
            
        self._filtered_gripper_q = (
            (1.0 - self.cfg.GRIPPER_SMOOTHING) * self._filtered_gripper_q
            + self.cfg.GRIPPER_SMOOTHING * q_clamped
        )

        self.current_gripper_targets = self._filtered_gripper_q.unsqueeze(-1)

        # update actions
        self.past_action = self.current_action.clone()
        self.current_action = torch.cat((self.robot_dof_targets[:, self.arm_joint_ids], self.current_gripper_targets), dim=-1)
        
        # randomized pushes to create instability for agent to recover from
        self._apply_push(torch.arange(self.num_envs, device=self.device))

    # actually tells the robots to reach the targets
    def _apply_action(self):
        # arm
        self._robot.set_joint_position_target(
            self.robot_dof_targets[:, self.arm_joint_ids],
            joint_ids=self.arm_joint_ids,
        )

        # gripper
        self._robot.set_joint_position_target(
            self.current_gripper_targets,
            joint_ids=torch.tensor([self.gripper_joint_ids[0]], device=self.device),
        )

    # --- post-physics step calls

    """ 
    RESEARCH QUESTION:
    Typical research or industry developments of RL policies use get dones as a simple safety measure and a failure/success checker
    A general evaluation of manipulation and other completion based tasks view a long terminaiton time as a negative sign
    Generally it means the agent isn't able to complete the in the allowed time
    My research challenges this idea using two main novel approaches:
    1. A truncation time that gradually increases as the agent's success rate increases
    2. No terminations except for PURE safety or unreachable success conditions, including success
    The hypothesis: forcing agents to train in long conditions with no success termination will:
    Increase the performance in sustained environments that don't exit/terminate and force the agent to continuely work
    Improve the reaction time to envrionment changes or failure
    """
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Notice how we DON'T terminate on success
        terminated = self._cubes_out_of_bounds()
        truncated = self.episode_length_buf >= self.adaptive_truncation_steps 

        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            L["dones/out_of_bounds"] = terminated.float().mean().item() # should report the training wide percent of OOB terminations
            L["dones/truncation_time"] = self.adaptive_truncation_steps.float().mean().item() # should report the traing wide average of truncation time
            L["dones/success_rate"] = (self.success_history.sum(dim=1) / self.success_count.clamp(min=1).float()).mean().item()

        return terminated, truncated
    
    # success check
    def _check_stack_success(self) -> torch.Tensor:
        xy_dist = torch.linalg.norm(self.cube_one_pos[:, :2] - self.cube_two_pos[:, :2], dim=-1)
        z_diff = self.cube_one_pos[:, 2] - self.cube_two_pos[:, 2]
        return (xy_dist < (0.0225 / 2)) & (z_diff > 0.02) & (z_diff < 0.03)

    def _cubes_out_of_bounds(self):

        terminated = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool
        )

        # cubes
        cubes = [self._cube_one, self._cube_two]
        for cube in cubes:
            pos = cube.data.root_pos_w - self.scene.env_origins
            out_x = (pos[:, 0] > self.cfg.x_bound) | (pos[:, 0] < -self.cfg.x_bound)
            out_y = (pos[:, 1] > self.cfg.y_bound) | (pos[:, 1] < -self.cfg.y_bound)
            out_z = (pos[:, 2] > self.cfg.z_bound) | (pos[:, 2] < 0.0)

            terminated |= (out_x | out_y | out_z)
        return terminated
    
    def _get_rewards(self) -> torch.Tensor:
        # refresh state tensors
        self._compute_intermediate_values()

        # Success
        step_success = self._check_stack_success()
        new_success = step_success & (~self.episode_success)
        self.episode_success |= step_success

        # Object Selection
        ee_pos = self.robot_ee_pos

        cube_one_pos = self.cube_one_pos
        cube_two_pos = self.cube_two_pos

        ee_cube_one_distance = torch.linalg.norm(
            cube_one_pos - ee_pos,
            dim=-1,
        )

        ee_cube_two_distance = torch.linalg.norm(
            cube_two_pos - ee_pos,
            dim=-1,
        )

        grasp_is_one = ee_cube_one_distance < ee_cube_two_distance

        grasp_cube_pos = torch.where(
            grasp_is_one.unsqueeze(-1),
            cube_one_pos,
            cube_two_pos,
        )

        target_cube_pos = torch.where(
            grasp_is_one.unsqueeze(-1),
            cube_two_pos,
            cube_one_pos,
        )

        grasp_cube_lin_vel = torch.where(
            grasp_is_one.unsqueeze(-1),
            self.cube_one_lin_vel,
            self.cube_two_lin_vel,
        )

        # Stack Target Pose
        stack_pose = target_cube_pos.clone()
        stack_pose[:, 2] += 0.0225

        # Distances
        ee_cube_distance = torch.linalg.norm(
            grasp_cube_pos - ee_pos,
            dim=-1,
        )

        ee_target_distance = torch.linalg.norm(
            target_cube_pos - ee_pos,
            dim=-1,
        )

        cube_to_stack_distance = torch.linalg.norm(
            stack_pose - grasp_cube_pos,
            dim=-1,
        )

        xy_stack_distance = torch.linalg.norm(
            grasp_cube_pos[:, :2] - target_cube_pos[:, :2],
            dim=-1,
        )

        height_error = torch.abs(
            grasp_cube_pos[:, 2] - stack_pose[:, 2]
        )

        # Reach + Lift
        reach_reward = 1.0 - torch.tanh(
            ee_cube_distance / 0.1
        )

        lift_height = grasp_cube_pos[:, 2]

        lifted = lift_height > 0.045

        lift_reward = (
            lifted.float()
            * (ee_cube_distance < 0.045).float()
        )

        # EE -> Target Rewards
        def shifted_sigmoid_reward(d, k=8.0, center=0.3):
            x = -k * (d - center)
            return 1.0 / (1.0 + torch.exp(-x))

        place_on_target_reward = (
            shifted_sigmoid_reward(ee_target_distance)
            * lifted.float()
            * (ee_cube_distance < 0.015).float()
        )

        place_on_target_fine_reward = (
            (1.0 - torch.tanh(ee_target_distance / 0.4))
            * lifted.float()
            * (ee_cube_distance < 0.015).float()
        )

        place_on_target_close_reward = (
            (1.0 - torch.tanh(ee_target_distance / 0.05))
            * (ee_target_distance < 0.2).float()
            * (ee_cube_distance < 0.015).float()
        )

        # Cube -> Stack Rewards
        stack_coarse_reward = (
            (1.0 - torch.tanh(cube_to_stack_distance / 0.10))
            * lifted.float()
        )

        stack_fine_reward = (
            (1.0 - torch.tanh(cube_to_stack_distance / 0.02))
            * lifted.float()
        )

        xy_alignment_reward = (
            (1.0 - torch.tanh(xy_stack_distance / 0.01))
            * lifted.float()
        )

        height_alignment_reward = (
            (1.0 - torch.tanh(height_error / 0.01))
            * lifted.float()
        )

        # Release Reward
        well_aligned = (
            (xy_stack_distance < 0.01)
            & (height_error < 0.01)
        )

        gripper_open = (
            self.gripper_width[:, 0] > 0.035
        )

        hand_away = (
            ee_cube_distance > 0.05
        )

        cube_stable = (
            torch.linalg.norm(
                grasp_cube_lin_vel,
                dim=-1,
            ) < 0.05
        )

        release_reward = (
            xy_alignment_reward
            * height_alignment_reward
            * well_aligned.float()
            * gripper_open.float()
            * hand_away.float()
            * cube_stable.float()
        )

        # Penalties
        action_diff_sq = torch.sum(torch.square(self.current_action - self.past_action),dim=-1,)
        joint_vel_sq = torch.sum(torch.square(self._robot.data.joint_vel),dim=-1,)

        self.count += 1
        progress = min(self.count / 2_000_000, 1.0,)

        action_penalty_rate = (1e-4 * (1.0 - progress) + self.cfg.action_penalty_final_rate * progress)
        joint_vel_penalty_rate = (1e-4 * (1.0 - progress) + self.cfg.joint_vel_penalty_final_rate * progress)

        # Final Reward
        reward = (
            self.cfg.reach_reward_weight * reach_reward
            + self.cfg.lift_reward_weight * lift_reward
            + (self.cfg.target_reward_weight* place_on_target_reward) ** 2
            + (self.cfg.target_fine_reward_weight * place_on_target_fine_reward) ** 2
            + (self.cfg.target_close_reward_weight * place_on_target_close_reward) ** 2
            + self.cfg.target_close_bonus_weight* place_on_target_close_reward
            + self.cfg.stack_coarse_reward_weight * stack_coarse_reward
            + self.cfg.stack_fine_reward_weight * stack_fine_reward
            + self.cfg.xy_alignment_reward_weight * xy_alignment_reward
            + self.cfg.height_alignment_reward_weight * height_alignment_reward
            + self.cfg.release_reward_weight * release_reward
            + self.cfg.success_reward_weight * new_success.float()
            - action_penalty_rate * action_diff_sq
            - joint_vel_penalty_rate * joint_vel_sq
        )

        # Logging
        if hasattr(self, "extras") and "log" in self.extras:
            L = self.extras["log"]
            L["reward/reach"] = float(reach_reward.mean().item())
            L["reward/lift"] = float(lift_reward.mean().item())
            L["reward/place_target"] = float(place_on_target_reward.mean().item())
            L["reward/place_target_fine"] = float(place_on_target_fine_reward.mean().item())
            L["reward/place_target_close"] = float(place_on_target_close_reward.mean().item())
            L["reward/stack_coarse"] = float(stack_coarse_reward.mean().item())
            L["reward/stack_fine"] = float(stack_fine_reward.mean().item())
            L["reward/xy_align"] = float(xy_alignment_reward.mean().item())
            L["reward/height_align"] = float(height_alignment_reward.mean().item())
            L["reward/release"] = float(release_reward.mean().item())
            L["reward/action_penalty"] = float(action_diff_sq.mean().item())
            L["reward/joint_vel_penalty"] = float(joint_vel_sq.mean().item())

        return reward
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # update the success history and trunction rates
        if env_ids is not None and env_ids.numel() > 0:
            self._update_success_history(env_ids=env_ids)
            self._update_adaptive_trunction(env_ids=env_ids)

        # robot state + rand pos noise
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        # should reset everything (if it doesn't reset gripper then try previous approach)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        # randomize the location of the cubes
        self._randomize_objects(env_ids=env_ids)
    
    def _randomize_objects(self, env_ids: torch.Tensor):
        for object in self.randomized_objects:
            num_resets = len(env_ids)
            new_pos = torch.zeros((num_resets, 3), device=self.device)
            new_pos[:, 0] = sample_uniform(-0.3, 0.3, (num_resets,), self.device)
            new_pos[:, 1] = sample_uniform(0.0, 0.35, (num_resets,), self.device)

            new_pos += self.scene.env_origins[env_ids]

            new_rot = torch.zeros((num_resets, 4), device=self.device)
            new_rot[:, 0] = 1.0

            root_state = torch.zeros((num_resets, 13), device=self.device)
            root_state[:, :3] = new_pos
            root_state[:, 3:7] = new_rot

            object.write_root_state_to_sim(root_state, env_ids=env_ids.to(torch.int32))

    def _apply_push(self, env_ids: torch.Tensor, frequency=1e-3, lin_magnitude=1e-1, ang_magnitude=1e-1):
        if env_ids.numel() == 0:
            return
            
        for obj in self.pushed_objects:
            # 1. FIXED TYPE CHECK: Cleanly ensure it's an asset type we can manipulate
            if not isinstance(obj, (RigidObject, Articulation)):
                continue

            num = len(env_ids)
            
            # Pick which envs the push should affect
            randoms = torch.rand((num,), device=self.device)
            mask = (randoms < frequency).float().unsqueeze(-1) # Shape: [num, 1]

            # Randomized forces and torques (centered around 0 by shifting scale down)
            forces = (torch.rand((num, 3), device=self.device) * 2.0 - 1.0) * lin_magnitude
            torques = (torch.rand((num, 3), device=self.device) * 2.0 - 1.0) * ang_magnitude
            
            # Scale and turn on/off pushes based on random frequency roll
            forces = forces * mask
            torques = torques * mask

            if isinstance(obj, RigidObject):
                # For basic objects like cubes, we apply directly to the object root
                obj.set_external_force_and_torque(
                    forces=forces,
                    torques=torques,
                    env_ids=env_ids,
                    is_global=True  # Interprets vectors relative to World Frame coordinates
                )
            elif isinstance(obj, Articulation):
                # For multi-body systems (like the robot arm), apply to its root link
                obj.set_external_force_and_torque(
                    forces=forces,
                    torques=torques,
                    body_ids=0, # Applies force vectors directly to the root base link
                    env_ids=env_ids,
                    is_global=True
                )

    # updates the success rates
    def _update_success_history(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        
        vals = self.episode_success[env_ids].float()
        idx = self.success_write_idx[env_ids]

        self.success_history[env_ids, idx] = vals
        self.success_write_idx[env_ids] = (idx + 1) % self.success_window
        self.success_count[env_ids] = torch.clamp(self.success_count[env_ids] + 1, max=self.success_window)

        # clear episode flag for next run
        self.episode_success[env_ids] = False

    # updates the reset times for the environments (uses success rates to do this)
    def _update_adaptive_trunction(self, env_ids:torch.Tensor):
        if env_ids.numel() == 0:
            return
        
        counts = self.success_count[env_ids].clamp(min=1)
        rolling_success = self.success_history[env_ids].sum(dim=1) / counts.float()
        rolling_success = torch.clamp(rolling_success, 0.0, 1.0)

        # exact 1x to 180x growth
        multiplier = torch.pow(
            torch.tensor(self.max_horizon_multiplier, device=self.device),
            rolling_success
        )

        new_steps = torch.round(self.base_episode_length * multiplier).long()
        new_steps = torch.clamp(new_steps, self.base_episode_length, self.base_episode_length * int(self.max_horizon_multiplier))

        self.adaptive_truncation_steps[env_ids] = new_steps

    def _get_observations(self) -> dict:
        # TODO: I likely need to normalize these
        # finger positions, end effector pose, flange pos, gripper width, joint velocities, cube poses 
        obs = torch.cat(
            (
                self._scale(self.left_pad_pos, self.cfg.pos_obs_scale), # finger positions (3)
                self._scale(self.right_pad_pos, self.cfg.pos_obs_scale), # (3)
                self._scale(self.robot_ee_pos, self.cfg.pos_obs_scale), # TCP (3)
                self._scale(self.robot_hand_pos, self.cfg.pos_obs_scale), # hand position for IK (3)
                self._scale(self.robot_hand_rot, self.cfg.quat_obs_scale), # shared with ee, for IK (4)
                self._scale(self.gripper_width, self.cfg.gripper_width_scale), # distance between pads (1)
                self._scale(self._robot.data.joint_vel[:, self.arm_joint_ids], self.cfg.joint_vel_scale), # the joint velocities (6)
                self._scale(self.cube_one_pos, self.cfg.pos_obs_scale), self._scale(self.cube_one_rot, self.cfg.quat_obs_scale), # (7)
                self._scale(self.cube_two_pos, self.cfg.pos_obs_scale), self._scale(self.cube_two_rot, self.cfg.quat_obs_scale), # (7)
                self._scale(self.cube_one_lin_vel, self.cfg.vel_obs_scale), self._scale(self.cube_one_ang_vel, self.cfg.ang_vel_obs_scale), # (6)
                self._scale(self.cube_two_lin_vel, self.cfg.vel_obs_scale), self._scale(self.cube_two_ang_vel, self.cfg.ang_vel_obs_scale), # (6)
                self._scale(self.ee_to_cube_one, self.cfg.pos_obs_scale),
                self._scale(self.ee_to_cube_two, self.cfg.pos_obs_scale),
                self._scale(self.cube_one_to_cube_two, self.cfg.pos_obs_scale),
                self._scale(self.actions[:, :6], self.cfg.action_obs_scale),
                self._scale(self.actions[:, 6:], self.cfg.gripper_action_obs_scale)
            ), dim=-1
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)} # max input values of 5

    def _scale(self, x: torch.Tensor, scale: float):
        return torch.clamp(x / scale, -5.0, 5.0)

    # auxiliary
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Global Body Pos
        # flange for the IK Controller
        self.robot_hand_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.hand_link_idx] - self.scene.env_origins[env_ids]
        self.robot_hand_rot[env_ids] = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        # Get finger positions and TCP
        self.left_pad_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.left_finger_link_idx] - self.scene.env_origins[env_ids]
        self.right_pad_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.right_finger_link_idx] - self.scene.env_origins[env_ids]
        self.robot_ee_pos[env_ids] = (self.left_pad_pos[env_ids] + self.right_pad_pos[env_ids]) / 2 # find center between two
        self.robot_ee_rot[env_ids] = self._robot.data.body_quat_w[env_ids, self.hand_link_idx] # extends from hand to get quat
        # gripper width
        width = torch.norm(self.left_pad_pos[env_ids] - self.right_pad_pos[env_ids], dim=-1, keepdim=True) # normalize into one scalar for raw diagonal distance
        self.gripper_width[env_ids] = width
        # cubes: env relative
        self.cube_one_pos[env_ids] = self._cube_one.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.cube_two_pos[env_ids] = self._cube_two.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.cube_one_rot[env_ids] = self._cube_one.data.root_quat_w[env_ids]
        self.cube_two_rot[env_ids] = self._cube_two.data.root_quat_w[env_ids]

        self.cube_one_lin_vel[env_ids] = self._cube_one.data.root_lin_vel_w[env_ids]
        self.cube_one_ang_vel[env_ids] = self._cube_one.data.root_ang_vel_w[env_ids]
        self.cube_two_lin_vel[env_ids] = self._cube_two.data.root_lin_vel_w[env_ids]
        self.cube_two_ang_vel[env_ids] = self._cube_two.data.root_ang_vel_w[env_ids]

        self.ee_to_cube_one[env_ids] = self.cube_one_pos[env_ids] - self.robot_ee_pos[env_ids]
        self.ee_to_cube_two[env_ids] = self.cube_two_pos[env_ids] - self.robot_ee_pos[env_ids]
        self.cube_one_to_cube_two[env_ids] = self.cube_two_pos[env_ids] - self.cube_one_pos[env_ids]

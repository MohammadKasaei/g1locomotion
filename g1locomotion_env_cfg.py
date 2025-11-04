# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp



import torch

# import isaaclab.envs.mdp as mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

import math
##
# Custom observation terms
##

# 

def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


##
# Scene definition
##


# from isaaclab_assets import G1_CFG  # isort: skip

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.25,
            ".*_knee_joint": -0.,
            ".*_ankle_pitch_joint": -0.25,

            ".*_elbow_pitch_joint": 1.57,
            "left_shoulder_roll_joint": 0.3,
            "left_shoulder_pitch_joint": 0.,
            
            "right_shoulder_roll_joint": -0.3,
            "right_shoulder_pitch_joint": 0.,
            
            "left_one_joint": .0,
            "right_one_joint": -.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", 
                ".*_hip_roll_joint",
                 ".*_hip_pitch_joint", 
                ".*_knee_joint",
                "torso_joint"
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.1,
                ".*_hip_roll_joint": 0.1,
                ".*_hip_pitch_joint": 0.1,
                ".*_knee_joint": 0.1,
                "torso_joint": 0.1,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint", 
                ".*_ankle_roll_joint"
            ],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0
            },
            damping={
                ".*_ankle_pitch_joint": 4.0,
                ".*_ankle_roll_joint": 4.0
            },
            armature={
                ".*_ankle_pitch_joint": 0.1,
                ".*_ankle_roll_joint": 0.1
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint"
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_pitch_joint": 40.0,
                ".*_elbow_roll_joint": 40.0,
                ".*_five_joint": 40.0,
                ".*_three_joint": 40.0,
                ".*_six_joint": 40.0,
                ".*_four_joint": 40.0,
                ".*_zero_joint": 40.0,
                ".*_one_joint": 40.0,
                ".*_two_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_pitch_joint": 10.0,
                ".*_elbow_roll_joint": 10.0,
                ".*_five_joint": 10.0,
                ".*_three_joint": 10.0,
                ".*_six_joint": 10.0,
                ".*_four_joint": 10.0,
                ".*_zero_joint": 10.0,
                ".*_one_joint": 10.0,
                ".*_two_joint": 10.0,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.1,
                ".*_shoulder_roll_joint": 0.1,
                ".*_shoulder_yaw_joint": 0.1,
                ".*_elbow_pitch_joint": 0.1,
                ".*_elbow_roll_joint": 0.1,
                ".*_five_joint": 0.1,
                ".*_three_joint": 0.1,
                ".*_six_joint": 0.1,
                ".*_four_joint": 0.1,
                ".*_zero_joint": 0.1,
                ".*_one_joint": 0.1,
                ".*_two_joint": 0.1,
            },
        ),
    },
)


  
@configclass
class G1LocoSceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""
    
    # Ground-plane
    terrain = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
   
    # # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # <-- update this line
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/defaultGroundPlane"],  # <-- update this line
    )
   
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # match all links of the Robot
        history_length=3,
        track_air_time=True
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-0, 0*math.pi)
            lin_vel_x=(1.0, 1.0), lin_vel_y=(-.0, .0), ang_vel_z=(-1.0, 1.0), heading=(-0*math.pi,0*math.pi)  
        ),
    )




@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # velocity_commands = ObsTerm(func=constant_commands)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions   = ObsTerm(func=mdp.last_action)
        walking_time = ObsTerm(func=mdp.walking_time)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "mass_distribution_params": (-0.1, 0.1),
    #         "operation": "add",
    #     },
    # )

    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(5.0, 10.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (-100.0, 100.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(5.0, 10.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Configuration for rewards."""
    # # Optuna optimized reward weights:
    #   track_lin_vel_xy_exp: 2.64722718723776
    #   track_ang_vel_z_exp: 0.8757072184783122
    #   symmetry_joint_motion: 0.10000000000000002
    #   feet_air_time: 0.1856814459918922
    #   feet_slide: -0.1087987299538326
    #   dof_pos_limits: -1.4956807119362252
    #   joint_deviation_hip: -0.27349065334738
    #   joint_deviation_arms: -0.15306600198911727
    #   joint_deviation_torso: -0.13080479722911748
    #   action_rate_l2: -0.01356353150914467
    #   dof_torques_l2: -1.1032243485864228e-05
    #   dof_acc_l2: -3.9570220618377686e-07
    #   flat_orientation_l2: -0.2655244814928903

     
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.64722718723776, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.8757072184783122, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    symmetry = RewTerm(
        func=mdp.symmetry_joint_motion,
        weight=0.10000000000000002,
        params={
            "left_cfg":  SceneEntityCfg("robot", joint_names=[
                "left_hip_yaw_joint",
                "left_hip_roll_joint",
                "left_hip_pitch_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ]),
            "right_cfg": SceneEntityCfg("robot", joint_names=[
                "right_hip_yaw_joint",
                "right_hip_roll_joint",
                "right_hip_pitch_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ]),
            # sign convention: +1 means same direction, -1 means mirrored
            "mirror_signs": [ +1.0, +1.0, -1.0, -1.0, -1.0, +1.0 ],
            "mode": "position_velocity",
        },
    )
     
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_symetry_biped,
        weight=0.1856814459918922,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.1,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1087987299538326,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.4956807119362252,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_pitch_joint", ".*ankle_roll_joint"])}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.27349065334738,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_.*", ".*_hip_roll_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15306600198911727,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight= -0.13080479722911748,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"])}
    )
    
    
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight= -0.01356353150914467)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.1032243485864228e-05)
    dof_acc_l2     = RewTerm(func=mdp.joint_acc_l2, weight=-3.9570220618377686e-07)
    
    # # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.body_orientation, weight=-0.2655244814928903, params={"limit_angle": 0.3})
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["head_link","torso_link"]), "threshold": 1.0},
    )
    body_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4},
    )

    # body_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"limit_angle": 0.3},
    # )


##
# Environment configuration
##


@configclass
class G1LocoEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    
   
    # Scene settings
    scene: G1LocoSceneCfg = G1LocoSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):

        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        # self.sim.physics_material = self.scene.terrain.physics_material
        # self.sim.device = args_cli.device
        self.episode_length_s = 20.0

        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        device = self.sim.device
        

        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

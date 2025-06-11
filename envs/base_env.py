import torch as th
import omnigibson as og
from omnigibson.robots import BaseRobot
import yaml
from omnigibson.envs.env_base import Environment
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.macros import gm
from omnigibson.utils.python_utils import create_class_from_registry_and_config
import copy
from pprint import pprint


class BaseEnv(Environment):
    """
    Custom environment class, inheriting from OmniGibson's Environment class,
    primarily adding dynamic object management functionality.
    """

    def __init__(self, config, in_vec_env=False, set_initial_camera=True):
        """
        Initialize the custom environment

        Args:
            configs (str or dict or list of str or dict): Path to config file or config dict
            in_vec_env (bool): Whether used in vectorized environment
            stabilize_scene (bool): Whether to execute scene stabilization steps after initialization
            set_initial_camera (bool): Whether to set initial camera position
        """

        # If config is a file path, load config file
        if isinstance(config, str) and config.endswith((".yaml", ".yml")):
            with open(config, "r") as f:
                config = yaml.safe_load(f)

        # Save configuration dict to environment
        self.cfg = copy.deepcopy(config)

        # Initialize dynamic objects list
        self.dynamic_objects = []

        # Process random table objects configuration
        if self.cfg.get("random_table_objects"):
            self.add_cluttered_objects_to_cfg()
            # debug
            pprint(self.cfg)
        # Print initialization message
        print("Creating environment, this may take some time...")

        # Call parent class initialization method
        super().__init__(configs=self.cfg, in_vec_env=in_vec_env)

        # Set camera position
        if set_initial_camera and gm.RENDER_VIEWER_CAMERA:
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor([0.450, 1.443, 1.678]),
                orientation=th.tensor([-0.075, 0.519, 0.843, -0.117]),
            )

        print("Environment creation complete!")

    def add_cluttered_objects_to_cfg(self):
        """
        Generate and merge cluttered objects to self.cfg
        """
        if not self.cfg.get("random_table_objects"):
            return {}

        # Generate cluttered object configuration
        # Import generate_cluttered_objects function
        from utils.table_object_generator import generate_cluttered_objects

        objects_cfg = generate_cluttered_objects(self.cfg)
        # Merge objects_cfg to self.cfg
        self.cfg["objects"].extend(objects_cfg)

    def reset(self, get_obs=True, **kwargs):
        """
        Reset the environment, including removing and re-adding dynamic objects

        Args:
            get_obs (bool): Whether to get observation result
            **kwargs: Other parameters passed to parent reset method

        Returns:
            Same return value as parent reset method
        """
        obs, _ = super().reset(get_obs=get_obs, **kwargs)

        self.set_robot_init_joint_positions()
        # self._stabilize_scene()
        return obs, {}

    def add_dynamic_objects(self, objects_cfg):
        """
        Generic method: Add dynamic objects to the environment and return the object list

        Args:
            objects_cfg (list): List of object configurations, each configuration is a dictionary containing parameters needed to create the object

        Returns:
            list: List of added dynamic objects
        """
        new_objects = []

        # Pause simulation to safely add objects
        is_playing = og.sim.is_playing()
        if is_playing:
            og.sim.pause()

        try:
            objects_instances = []

            # Create object instances
            for obj_cfg in objects_cfg:
                # Ensure config has type field
                if "type" not in obj_cfg:
                    print(
                        f"Warning: Missing type field in configuration, cannot create object"
                    )
                    continue

                # Copy config to avoid modifying original config
                cfg_copy = obj_cfg.copy()

                # Store position and orientation
                position = cfg_copy.pop("position", None)
                orientation = cfg_copy.pop("orientation", None)

                # Use create_class_from_registry_and_config to create object
                # This supports different types of objects, including DatasetObject and PrimitiveObject
                try:
                    obj = create_class_from_registry_and_config(
                        cls_name=cfg_copy["type"],
                        cls_registry=REGISTERED_OBJECTS,
                        cfg=cfg_copy,
                        cls_type_descriptor="object",
                    )
                    objects_instances.append((obj, position, orientation))
                except Exception as e:
                    print(f"Error creating object: {e}")
                    continue

            # Add all objects in batch
            for obj, position, orientation in objects_instances:
                try:
                    self.scene.add_object(obj)
                    # Set position
                    if position is not None or orientation is not None:
                        obj.set_position_orientation(
                            position=position, orientation=orientation, frame="scene"
                        )
                    new_objects.append(obj)
                except Exception as e:
                    print(f"Error adding object: {e}")
        finally:
            # Restore simulation state
            if is_playing:
                og.sim.play()
                # Execute one physics step to stabilize objects
                og.sim.step_physics()

        # Add newly added objects to dynamic objects list
        self.dynamic_objects.extend(new_objects)
        print(f"Successfully added {len(new_objects)} dynamic objects")

        return new_objects

    def remove_dynamic_objects(self):
        """
        Remove all dynamically added objects from the environment
        """
        objects_to_remove = []

        # Collect all objects to remove
        for obj in self.dynamic_objects:
            if obj is not None and hasattr(obj, "scene") and obj.scene is not None:
                objects_to_remove.append(obj)

        # Clear dynamic objects list
        self.dynamic_objects = []

        # If there are objects to remove, use batch_remove_objects
        if objects_to_remove:
            try:
                # Pause simulation to safely remove objects
                is_playing = og.sim.is_playing()
                if is_playing:
                    og.sim.pause()

                # Remove all objects in batch
                og.sim.batch_remove_objects(objects_to_remove)

                # Restore simulation state
                if is_playing:
                    og.sim.play()
            except Exception as e:
                print(f"Error in batch remove objects: {e}")

    def set_robot_init_joint_positions(self, joint_positions=None):
        """
        Set the initial joint positions of the robot

        Args:
            joint_positions (dict or tensor, optional): Joint position mapping or tensor.
                                                      If None, will set default head orientation.
        """
        if not self.robots:
            print("No robot in environment, cannot set joint positions")
            return

        robot: BaseRobot = self.robots[0]

        if joint_positions is None:
            # Default to set head orientation joint
            joint_positions = robot.get_joint_positions()
            head_pan_idx = list(robot.joints.keys()).index("head_2_joint")
            joint_positions[head_pan_idx] = -1

        # Set joint positions
        robot.set_joint_positions(joint_positions)
        print("Set robot initial joint positions")

        # Save current joint positions as default, so they'll be used in subsequent resets
        robot.reset_joint_pos = joint_positions.clone()
        print("Set current joint positions as default reset positions")

        return joint_positions

    def _stabilize_scene(self, steps=30):
        """
        Execute multiple zero-action steps to stabilize the scene

        Args:
            steps (int): Number of steps to execute
        """
        print("Waiting for scene to stabilize...")
        robot = self.robots[0]
        for _ in range(steps):
            # Create zero action
            zero_action = th.zeros(robot.action_dim)
            self.step(zero_action)
        print("Scene stabilization complete")

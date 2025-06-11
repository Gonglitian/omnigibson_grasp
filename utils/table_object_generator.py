"""
Table Object Generator Module

This module provides functionality to generate random objects on table surfaces
in OmniGibson environments with proper positioning and orientation.

Features:
- Grid-based positioning system for systematic object placement
- Configurable object density and padding
- Support for table rotation and world coordinate transformation
- Type-safe implementation with comprehensive error handling
- Backward compatibility with legacy function interfaces

Example Usage:
    # Using the new class-based interface (recommended)
    generator = TableObjectGenerator(env_config)
    object_configs = generator.generate()
    
    # Using the legacy function interface (for backward compatibility)
    object_configs = generate_cluttered_objects(env_config)

Configuration Format:
    The environment configuration should include:
    {
        "random_table_objects": {
            "table_name": "dining_table",
            "table_length": 1.5,
            "table_width": 0.8,
            "table_height": 0.05,
            "categories": ["apple", "bottle", "bowl"],
            "num_objects": 10,
            "grid_size": 0.1,
            "occupancy_rate": 0.6,
            "padding": 0.1,
            "random_models": true
        },
        "objects": [
            {
                "name": "dining_table",
                "position": [1.0, 0.5, 0.8],
                "orientation": [0, 0, 0, 1]
            }
        ]
    }
"""

import random
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import torch as th

from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)
from omnigibson.utils.transform_utils import random_quaternion, quat_apply


class TableObjectGenerator:
    """Generator for placing objects on table surfaces with grid-based positioning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the table object generator.
        
        Args:
            config: Environment configuration containing table and object settings
        """
        self.config = config
        self.func_config = config.get("random_table_objects", {})
        
    def generate(self) -> List[Dict[str, Any]]:
        """
        Generate object configurations for placement on table surface.
        
        Returns:
            List of object configurations ready for environment setup
        """
        # Validate configuration
        if not self._validate_config():
            return []
            
        # Read table information
        table_info = self._read_table_info()
        if table_info is None:
            return []
            
        # Generate grid positions on table surface
        positions = self._generate_positions(table_info)
        if positions.size(0) == 0:
            print("Warning: No valid positions generated on table surface")
            return []
            
        # Create object configurations
        return self._create_objects_config(positions)
    
    def _validate_config(self) -> bool:
        """
        Validate the configuration for object generation.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.func_config:
            print("Error: 'random_table_objects' configuration not found")
            return False
            
        required_fields = ["table_name", "table_length", "table_width", "table_height"]
        for field in required_fields:
            if field not in self.func_config:
                print(f"Error: Required field '{field}' not found in configuration")
                return False
                
        if "objects" not in self.config:
            print("Error: No objects found in environment configuration")
            return False
            
        return True
    
    def _read_table_info(self) -> Optional[Dict[str, Any]]:
        """
        Extract table information from configuration.
        
        Returns:
            Dictionary containing table parameters or None if not found
        """
        table_name = self.func_config["table_name"]
        
        # Search for table in objects list
        for obj in self.config["objects"]:
            if obj.get("name") == table_name:
                return {
                    "length": self.func_config["table_length"],
                    "width": self.func_config["table_width"], 
                    "height": self.func_config["table_height"],
                    "position": obj["position"],
                    "orientation": obj.get("orientation")
                }
        
        print(f"Error: Table '{table_name}' not found in configuration")
        return None
    
    def _generate_positions(self, table_info: Dict[str, Any]) -> th.Tensor:
        """
        Generate grid positions on table surface.
        
        Args:
            table_info: Dictionary containing table parameters
            
        Returns:
            Tensor of 3D positions on table surface
        """
        # Get configuration parameters
        grid_size = self.func_config.get("grid_size", 0.1)
        occupancy_rate = self.func_config.get("occupancy_rate", 0.5)
        padding = self.func_config.get("padding", 0.1)
        
        # Calculate usable table area
        usable_length = table_info["length"] - 2 * padding
        usable_width = table_info["width"] - 2 * padding
        
        if usable_length <= 0 or usable_width <= 0:
            print("Error: Table dimensions too small for given padding")
            return th.empty(0, 3)
        
        # Calculate grid dimensions
        grid_cols = int(usable_length / grid_size)
        grid_rows = int(usable_width / grid_size)
        total_cells = grid_cols * grid_rows
        num_positions = int(total_cells * occupancy_rate)
        
        print(f"Table grid: {grid_cols}x{grid_rows} = {total_cells} cells")
        print(f"Using {num_positions} positions ({occupancy_rate:.1%} occupancy)")
        
        if num_positions == 0:
            return th.empty(0, 3)
        
        # Generate grid positions
        positions = self._create_grid_positions(
            grid_cols, grid_rows, grid_size, usable_length, usable_width
        )
        
        # Add random offsets within grid cells
        positions = self._add_random_offsets(positions, grid_size)
        
        # Apply table rotation if specified
        if table_info["orientation"] is not None:
            positions = self._apply_rotation(positions, table_info["orientation"])
        
        # Transform to world coordinates
        table_position = th.tensor(table_info["position"], dtype=th.float32)
        positions = positions + table_position
        
        # Set height to table surface
        positions[:, 2] = table_position[2] + table_info["height"]
        
        # Randomly select subset of positions
        indices = th.randperm(positions.size(0))[:num_positions]
        return positions[indices]
    
    def _create_grid_positions(self, cols: int, rows: int, grid_size: float, 
                             length: float, width: float) -> th.Tensor:
        """Create regular grid positions centered on table."""
        # Calculate grid starting offsets (to center the grid)
        offset_x = -length / 2 + grid_size / 2
        offset_y = -width / 2 + grid_size / 2
        
        # Create coordinate arrays
        x_coords = th.arange(cols, dtype=th.float32) * grid_size + offset_x
        y_coords = th.arange(rows, dtype=th.float32) * grid_size + offset_y
        
        # Create meshgrid and flatten
        grid_x, grid_y = th.meshgrid(x_coords, y_coords, indexing="xy")
        positions = th.stack([
            grid_x.flatten(),
            grid_y.flatten(), 
            th.zeros_like(grid_x.flatten())
        ], dim=1)
        
        return positions
    
    def _add_random_offsets(self, positions: th.Tensor, grid_size: float) -> th.Tensor:
        """Add random offsets within grid cells."""
        # Generate random offsets within [-0.5, 0.5] * grid_size
        offsets = (th.rand_like(positions) - 0.5) * grid_size
        offsets[:, 2] = 0.0  # No Z offset
        return positions + offsets
    
    def _apply_rotation(self, positions: th.Tensor, orientation: Any) -> th.Tensor:
        """Apply table rotation to positions."""
        try:
            if not isinstance(orientation, th.Tensor):
                orientation = th.tensor(orientation, dtype=th.float32)
            return quat_apply(orientation, positions)
        except Exception as e:
            print(f"Warning: Failed to apply rotation: {e}")
            return positions
    
    def _create_objects_config(self, positions: th.Tensor) -> List[Dict[str, Any]]:
        """
        Create object configurations for given positions.
        
        Args:
            positions: Tensor of 3D positions for object placement
            
        Returns:
            List of object configurations
        """
        # Get object generation parameters
        categories = self.func_config.get("categories", [])
        num_objects = self.func_config.get("num_objects", len(categories))
        random_models = self.func_config.get("random_models", True)
        
        if not categories:
            print("Warning: No object categories specified")
            return []
        
        # Calculate objects per category
        object_counts = self._calculate_object_counts(categories, num_objects, positions.size(0))
        
        # Generate object configurations
        objects_config = []
        position_idx = 0
        
        for category, count in zip(categories, object_counts):
            for i in range(count):
                if position_idx >= positions.size(0):
                    break
                    
                obj_config = self._create_single_object_config(
                    category, positions[position_idx], i, random_models
                )
                
                if obj_config:
                    objects_config.append(obj_config)
                    position_idx += 1
        
        print(f"Generated {len(objects_config)} object configurations")
        return objects_config
    
    def _calculate_object_counts(self, categories: List[str], num_objects: Union[int, List[int]], 
                                max_positions: int) -> List[int]:
        """Calculate how many objects to create per category."""
        if isinstance(num_objects, list):
            if len(num_objects) != len(categories):
                print("Warning: num_objects length doesn't match categories length")
                # Adjust list length
                if len(num_objects) < len(categories):
                    num_objects.extend([1] * (len(categories) - len(num_objects)))
                else:
                    num_objects = num_objects[:len(categories)]
            object_counts = num_objects
        else:
            # Distribute total count evenly across categories
            base_count = num_objects // len(categories)
            remainder = num_objects % len(categories)
            object_counts = [base_count] * len(categories)
            # Distribute remainder
            for i in range(remainder):
                object_counts[i] += 1
        
        # Ensure we don't exceed available positions
        total_requested = sum(object_counts)
        if total_requested > max_positions:
            scale_factor = max_positions / total_requested
            object_counts = [max(1, int(count * scale_factor)) for count in object_counts]
            # Adjust to exact count
            while sum(object_counts) > max_positions:
                max_idx = object_counts.index(max(object_counts))
                object_counts[max_idx] -= 1
        
        return object_counts
    
    def _create_single_object_config(self, category: str, position: th.Tensor, 
                                   index: int, random_models: bool) -> Optional[Dict[str, Any]]:
        """Create configuration for a single object."""
        # Get available models for category
        try:
            available_models = get_all_object_category_models(category)
        except Exception as e:
            print(f"Warning: Failed to get models for category '{category}': {e}")
            return None
            
        if not available_models:
            print(f"Warning: No models found for category '{category}'")
            return None
        
        # Select model
        model = random.choice(available_models) if random_models else available_models[0]
        
        return {
            "type": "DatasetObject",
            "name": f"{category}_{index + 1}",
            "category": category,
            "model": model,
            "fixed_base": False,
            "position": position.tolist(),
            "orientation": self._generate_random_orientation()
        }
    
    def _generate_random_orientation(self, axis_aligned: bool = False) -> List[float]:
        """
        Generate random orientation quaternion.
        
        Args:
            axis_aligned: If True, only rotate around Z-axis to keep objects upright
            
        Returns:
            Quaternion as [x, y, z, w] list
        """
        if axis_aligned:
            # Only rotate around Z-axis
            angle = random.uniform(0, 2 * math.pi)
            return [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
        else:
            # Completely random orientation
            return random_quaternion(1).tolist()[0]


# Convenience function for backward compatibility
def generate_cluttered_objects(env_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate cluttered objects configuration for table surface.
    
    Args:
        env_cfg: Environment configuration dictionary
        
    Returns:
        List of object configurations
    """
    generator = TableObjectGenerator(env_cfg)
    return generator.generate()


# Legacy functions for backward compatibility
def read_table_info(cfg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], 
                                                 Optional[float], Optional[List[float]], 
                                                 Optional[List[float]]]:
    """Legacy function - use TableObjectGenerator instead."""
    generator = TableObjectGenerator(cfg)
    table_info = generator._read_table_info()
    if table_info is None:
        return None, None, None, None, None
    return (table_info["length"], table_info["width"], table_info["height"],
            table_info["position"], table_info["orientation"])


def generate_grid_positions(table_length: float, table_width: float, table_height: float,
                          table_position: Optional[List[float]] = None,
                          table_orientation: Optional[List[float]] = None,
                          grid_size: float = 0.1, occupancy_rate: float = 0.5,
                          padding: float = 0.1) -> Tuple[th.Tensor, th.Tensor]:
    """Legacy function - use TableObjectGenerator instead."""
    # Create minimal config for compatibility
    config = {
        "random_table_objects": {
            "table_name": "table",
            "table_length": table_length,
            "table_width": table_width, 
            "table_height": table_height,
            "grid_size": grid_size,
            "occupancy_rate": occupancy_rate,
            "padding": padding
        },
        "objects": [{"name": "table", "position": table_position, "orientation": table_orientation}]
    }
    
    generator = TableObjectGenerator(config)
    table_info = generator._read_table_info()
    if table_info is None:
        return th.empty(0, 3), th.empty(0, 3)
        
    positions = generator._generate_positions(table_info)
    return positions, positions  # Return same tensor twice for compatibility 
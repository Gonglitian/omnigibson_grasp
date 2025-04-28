import random
import math
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)
from omnigibson.utils.transform_utils import random_quaternion


def read_table_info(cfg):
    func_cfg = cfg["random_table_objects"]
    for obj in cfg["objects"]:
        if obj["name"] == func_cfg["table_name"]:
            table_position = obj["position"]
            table_orientation = obj["orientation"]
            # get length, width, height from cfg
            table_length = func_cfg["table_length"]
            table_width = func_cfg["table_width"]
            table_height = func_cfg["table_height"]
        else:
            print(
                f"Error: Table {obj['name']} not found in cfg, will use default table"
            )
            return None, None, None, None, None
    return table_length, table_width, table_height, table_position, table_orientation


def generate_grid_positions(
    table_length,
    table_width,
    table_height,
    table_position=None,
    table_orientation=None,
    grid_size=0.1,
    occupancy_rate=0.5,
    padding=0.1,
):
    """
    Generate grid positions on the table surface, based on fixed grid size

    Args:
        table_length (float): Table length
        table_width (float): Table width
        table_height (float): Table height
        table_orientation (torch.Tensor): Table orientation quaternion [x, y, z, w]
        grid_size (float): Fixed size of each grid
        occupancy_rate (float): Object density
        padding (float): Edge padding to avoid placing objects on table edges
    Returns:
        tuple: (torch.Tensor, int) - Generated positions and number of available positions
    """
    import torch as th
    from omnigibson.utils.transform_utils import quat_apply

    # Calculate usable area
    usable_width = table_length - 2 * padding
    usable_depth = table_width - 2 * padding

    # Calculate usable grid count
    grid_cols = int(usable_width / grid_size)
    grid_rows = int(usable_depth / grid_size)
    total_grid_cells = grid_cols * grid_rows
    print(f"total_grid_cells: {total_grid_cells}")
    # Calculate available positions based on occupancy rate
    num_positions = int(total_grid_cells * occupancy_rate)

    print(f"Table surface size: {usable_width} x {usable_depth}")
    print(f"Grid size: {grid_size} x {grid_size}")
    print(f"Grid count: {grid_cols} x {grid_rows} = {total_grid_cells}")
    print(f"Occupancy rate: {occupancy_rate}")
    print(f"Number of available positions: {num_positions}")

    # Calculate starting offset (to center the grid)
    start_offset_x = -usable_width / 2 + grid_size / 2
    start_offset_y = -usable_depth / 2 + grid_size / 2

    # Create row and column index grid
    i_indices = th.arange(grid_cols, dtype=th.float32)
    j_indices = th.arange(grid_rows, dtype=th.float32)

    # Use meshgrid to create 2D grid
    grid_x, grid_y = th.meshgrid(
        start_offset_x + i_indices * grid_size,
        start_offset_y + j_indices * grid_size,
        indexing="xy",
    )

    # Flatten the grid and combine into 3D coordinates (x, y, 0)
    all_grid_positions = th.stack(
        [grid_x.flatten(), grid_y.flatten(), th.zeros_like(grid_x.flatten())], dim=1
    )

    # Generate random offsets to distribute positions randomly within their grids
    random_offsets = th.rand(all_grid_positions.shape, dtype=th.float32)
    random_offsets[:, 0] = (
        random_offsets[:, 0] - 0.5
    ) * grid_size  # Limit random offset to half the grid size
    random_offsets[:, 1] = (random_offsets[:, 1] - 0.5) * grid_size
    random_offsets[:, 2] = 0.0  # z offset is 0

    # Apply random offsets
    relative_positions = all_grid_positions + random_offsets

    # If table orientation is provided, apply rotation
    if table_orientation is not None:
        try:
            # Ensure orientation is a tensor
            if not isinstance(table_orientation, th.Tensor):
                table_orientation = th.tensor(table_orientation, dtype=th.float32)

            # Use quat_apply function to batch rotate all points
            relative_positions = quat_apply(table_orientation, relative_positions)
        except Exception as e:
            print(f"Failed to apply rotation: {e}, will use unrotated coordinates")

    # Add surface center coordinates to get final positions
    final_positions = relative_positions + table_position[2]

    # Add height
    final_positions[:, 2] = final_positions[:, 2] + table_height

    # Randomly shuffle position order
    indices = th.randperm(final_positions.shape[0])
    final_positions = final_positions[indices][:num_positions]

    print(f"Generated {final_positions.shape[0]} positions")

    return final_positions, all_grid_positions


def random_orientation(axis_aligned=False):
    """
    Generate random orientation quaternion

    Args:
        axis_aligned (bool): If True, only rotate randomly around Z axis to keep objects upright
                          If False, generate completely random orientation

    Returns:
        list: Quaternion in [x, y, z, w] format
    """
    import torch as th

    if axis_aligned:
        # Only rotate randomly around Z axis (0, 0, sin(θ/2), cos(θ/2))
        angle = random.uniform(0, 2 * math.pi)  # Random angle
        return [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
    else:
        # Completely random orientation, using omnigibson's provided function
        return random_quaternion(1).tolist()[0]


def generate_cluttered_objects(
    env_cfg: dict,
):
    """
    Generate multiple object configurations to make the table cluttered

    Args:
        categories (list): List of object categories, if None, will randomly select common object categories
        num_objects (int or list): Number of objects, if None and categories is a list, will be the length of categories;
                               if list, should match the length of categories list, indicating number of objects per category
        random_models (bool): Whether to randomly select models, otherwise select the first available model
        env (og.Environment): OmniGibson environment, used to get the actual table dimensions
        table_name (str): Name of the table
        grid_size (float): Fixed size of each grid
        return_cfg (bool): If True, only return configuration without adding objects to environment
        **kwargs: Additional parameters, such as padding, etc.

    Returns:
        list: List containing multiple object configurations, or list of objects added to environment
    """
    # get parameters from cfg
    func_cfg = env_cfg["random_table_objects"]
    # config for generate object configurations
    categories = func_cfg.get("categories", None)
    num_objects = func_cfg.get("num_objects", None)
    random_models = func_cfg.get("random_models", True)
    grid_size = func_cfg.get("grid_size", 0.1)
    padding = func_cfg.get("padding", 0.1)
    occupancy_rate = func_cfg.get("occupancy_rate", 0.5)
    auto_supplement = func_cfg.get("auto_supplement", False)

    # info for table
    table_length, table_width, table_height, table_position, table_orientation = (
        read_table_info(env_cfg)
    )

    # Process object count configuration
    if isinstance(num_objects, list):
        # If num_objects is a list, ensure it matches categories length
        if len(num_objects) != len(categories):
            print(
                f"Warning: num_objects length({len(num_objects)}) does not match categories length({len(categories)})"
            )
            if len(num_objects) < len(categories):
                # If num_objects is shorter, pad with default value 1
                num_objects = num_objects + [1] * (len(categories) - len(num_objects))
            else:
                # If num_objects is longer, truncate extra items
                num_objects = num_objects[: len(categories)]

        # Calculate expected total objects
        expected_total_objects = sum(num_objects)
    elif isinstance(num_objects, int):
        # If num_objects is a single number, it represents total count
        expected_total_objects = num_objects
        # Create an evenly distributed list
        num_per_category = expected_total_objects // len(categories)
        remainder = expected_total_objects % len(categories)
        num_objects = [num_per_category] * len(categories)
        # Distribute remainder to first few categories
        for i in range(remainder):
            num_objects[i] += 1
    else:
        # Default to 1 object per category
        num_objects = [1] * len(categories)
        expected_total_objects = len(categories)

    # Get table bounding box information and number of available positions
    positions = None

    # Get grid positions
    positions, all_grid_positions = generate_grid_positions(
        table_length,
        table_width,
        table_height,
        table_position,
        table_orientation,
        grid_size,
        occupancy_rate,
        padding,
    )
    available_positions = positions.shape[0]
    # Adjust object count based on relationship between available_positions and expected_total_objects
    if available_positions < expected_total_objects:
        # If available positions are fewer than expected total objects, proportionally truncate each category's count
        print(
            f"Warning: Available positions ({available_positions}) fewer than expected total objects ({expected_total_objects}), will truncate object count"
        )
        scale_factor = available_positions / expected_total_objects
        new_num_objects = []
        total_allocated = 0

        # First allocate integer parts
        for n in num_objects:
            allocated = int(n * scale_factor)
            new_num_objects.append(allocated)
            total_allocated += allocated

        # Allocate remaining positions
        remaining = available_positions - total_allocated
        i = 0
        while remaining > 0 and i < len(new_num_objects):
            new_num_objects[i] += 1
            remaining -= 1
            i += 1

        num_objects = new_num_objects
        expected_total_objects = available_positions
    elif available_positions > expected_total_objects and auto_supplement:
        # If available positions are more than expected total objects, and auto-supplement is enabled, can increase object count
        # Note: Only supplement if auto_supplement is explicitly enabled in configuration
        print(
            f"Available positions ({available_positions}) more than expected total objects ({expected_total_objects}), can add more objects"
        )
        extra_positions = available_positions - expected_total_objects

        # Proportionally increase each category's count
        if extra_positions > 0:
            original_total = sum(num_objects)
            for i in range(len(num_objects)):
                extra = int(extra_positions * (num_objects[i] / original_total))
                num_objects[i] += extra
                extra_positions -= extra

            # Distribute remaining positions
            i = 0
            while extra_positions > 0 and i < len(num_objects):
                num_objects[i] += 1
                extra_positions -= 1
                i += 1

            expected_total_objects = available_positions

    # Expand categories and count list
    expanded_categories = []
    for i, (category, count) in enumerate(zip(categories, num_objects)):
        expanded_categories.extend([category] * count)
    print(f"expanded_categories: {expanded_categories}")
    # Generate object configurations
    objects_cfg = []
    num_positions_used = 0
    for i, category in enumerate(expanded_categories):
        # Ensure not exceeding available positions
        if num_positions_used >= available_positions:
            break

        # Get all available models for this category
        available_models = get_all_object_category_models(category)

        if len(available_models) == 0:
            print(
                f"Warning: Category {category} does not exist or has no available models, skipping"
            )
            continue

        # Select model
        if random_models:
            model = random.choice(available_models)
        else:
            model = available_models[0]

        # Set position
        if num_positions_used < positions.shape[0]:
            # Use pre-generated grid positions
            position = positions[
                num_positions_used
            ].tolist()  # Convert to list to adapt to DatasetObject
            num_positions_used += 1
        else:
            # This should not happen as we've already limited the loop count
            print(
                f"Warning: Position index {num_positions_used} exceeds pre-generated positions range {positions.shape[0]}, skipping"
            )
            continue

        # Create object configuration
        objects_cfg.append(
            {
                "type": "DatasetObject",
                "name": f"{category}_{i+1}",
                "category": category,
                "model": model,
                "fixed_base": False,
                "position": position,
                "orientation": random_orientation(),
            }
        )

    return objects_cfg

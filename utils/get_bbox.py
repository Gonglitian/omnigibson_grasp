# get obj bbox thru name
def get_obj_bbox(env, obj_name):
    """
    Get object bounding box information

    Args:
        env (og.Environment): OmniGibson environment
        obj_name (str): Name of the object

    Returns:
        tuple: (bbox_center, bbox_extent, obj_height, obj_orientation) Object's bounding box center, size, height and orientation
    """
    # Get the object
    obj = None
    for obj in env.scene.objects:
        if obj.name == obj_name:
            table = obj
            break

    if obj is None:
        print(f"Warning: Object named {obj_name} not found, cannot get bounding box")
        return None, None, None, None

    # Use get_base_aligned_bbox to get the object's bounding box information
    # Parameter xy_aligned=True ensures the bounding box is aligned with the XY plane
    bbox_center_world, bbox_orn_world, bbox_extent, bbox_center_local = (
        table.get_base_aligned_bbox(xy_aligned=True)
    )

    # Get the object's position and orientation
    pos, orn = obj.get_position_orientation()

    # Calculate the actual height of the object
    obj_height = bbox_center_world[2] + bbox_extent[2] / 2

    return bbox_center_world, bbox_extent, obj_height, orn

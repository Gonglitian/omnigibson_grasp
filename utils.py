import random
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)
import copy
import math
from omnigibson.utils.transform_utils import random_quaternion


def get_table_bbox(env, table_name="table"):
    """
    获取桌子的包围盒信息

    参数:
        env (og.Environment): OmniGibson环境
        table_name (str): 桌子的名称

    返回:
        tuple: (bbox_center, bbox_extent, table_height, table_orientation) 桌子的包围盒中心、尺寸、高度和朝向
    """
    # 获取桌子对象
    table = None
    for obj in env.scene.objects:
        if obj.name == table_name:
            table = obj
            break

    if table is None:
        print(f"警告: 未找到名为 {table_name} 的桌子，无法获取包围盒")
        return None, None, None, None

    # 使用get_base_aligned_bbox获取桌子的包围盒信息
    # 参数xy_aligned=True确保包围盒与XY平面对齐
    bbox_center_world, bbox_orn_world, bbox_extent, bbox_center_local = table.get_base_aligned_bbox(xy_aligned=True)

    # 获取桌子的位置和朝向
    pos, orn = table.get_position_orientation()

    # 计算桌子的实际高度
    table_height = bbox_center_world[2] + bbox_extent[2] / 2

    print(f"桌子包围盒中心: {bbox_center_world}")
    print(f"桌子包围盒尺寸: {bbox_extent}")
    print(f"桌子高度: {table_height}")
    print(f"桌子朝向: {orn}")

    return bbox_center_world, bbox_extent, table_height, orn


def generate_grid_positions(
    bbox_center, bbox_extent, table_orientation=None, grid_size=0.1, occupancy_rate=0.5, padding=0.1
):
    """
    在桌面上生成网格状的位置点，基于固定网格大小

    参数:
        bbox_center (torch.Tensor): 包围盒中心坐标 [x, y, z]
        bbox_extent (torch.Tensor): 包围盒尺寸 [width, depth, height]
        table_orientation (torch.Tensor): 桌子的朝向四元数 [x, y, z, w]
        grid_size (float): 每个网格的固定大小
        occupancy_rate (float): 物品密度
        padding (float): 边缘填充，避免物体放在桌子边缘
    返回:
        tuple: (torch.Tensor, int) - 生成的位置点和可用位置数量
    """
    import torch as th
    from omnigibson.utils.transform_utils import quat_apply

    # 确保输入为张量
    if not isinstance(bbox_center, th.Tensor):
        bbox_center = th.tensor(bbox_center, dtype=th.float32)
    if not isinstance(bbox_extent, th.Tensor):
        bbox_extent = th.tensor(bbox_extent, dtype=th.float32)

    # 计算可用面积
    usable_width = bbox_extent[0] - 2 * padding
    usable_depth = bbox_extent[1] - 2 * padding

    # 计算可用网格数量
    grid_cols = int(usable_width / grid_size)
    grid_rows = int(usable_depth / grid_size)
    total_grid_cells = grid_cols * grid_rows
    print(f"total_grid_cells: {total_grid_cells}")
    # 根据占用率计算可用位置数量
    available_positions = int(total_grid_cells * occupancy_rate)

    print(f"桌面尺寸: {usable_width} x {usable_depth}")
    print(f"网格大小: {grid_size} x {grid_size}")
    print(f"网格数量: {grid_cols} x {grid_rows} = {total_grid_cells}")
    print(f"占用率: {occupancy_rate}")
    print(f"可用位置数量: {available_positions}")

    # 计算桌子中心坐标
    table_center = bbox_center.clone()
    # 调整Z高度到桌面上方
    table_center[2] = bbox_center[2] + bbox_extent[2] / 2 + 0.1  # 桌面高度加上一点偏移

    # 计算起始偏移量（使网格居中）
    start_offset_x = -usable_width / 2 + grid_size / 2
    start_offset_y = -usable_depth / 2 + grid_size / 2

    # 创建行列索引网格
    i_indices = th.arange(grid_cols, dtype=th.float32)
    j_indices = th.arange(grid_rows, dtype=th.float32)

    # 使用meshgrid创建2D网格
    grid_x, grid_y = th.meshgrid(
        start_offset_x + i_indices * grid_size, start_offset_y + j_indices * grid_size, indexing="xy"
    )

    # 将网格展平并组合成3D坐标 (x, y, 0)
    grid_positions = th.stack([grid_x.flatten(), grid_y.flatten(), th.zeros_like(grid_x.flatten())], dim=1)

    # 生成随机偏移，使位置在各自网格内随机分布
    random_offsets = th.rand(grid_positions.shape, dtype=th.float32)
    random_offsets[:, 0] = (random_offsets[:, 0] - 0.5) * grid_size  # 限制随机偏移不超过网格的一半
    random_offsets[:, 1] = (random_offsets[:, 1] - 0.5) * grid_size
    random_offsets[:, 2] = 0.0  # z偏移为0

    # 应用随机偏移
    relative_positions = grid_positions + random_offsets

    # 如果提供了桌子朝向，应用旋转
    if table_orientation is not None:
        try:
            # 确保orientation是张量
            if not isinstance(table_orientation, th.Tensor):
                table_orientation = th.tensor(table_orientation, dtype=th.float32)

            # 直接使用quat_apply函数批量旋转所有点
            relative_positions = quat_apply(table_orientation, relative_positions)
        except Exception as e:
            print(f"应用旋转失败: {e}，将使用未旋转的坐标")

    # 添加表面中心坐标得到最终位置
    final_positions = relative_positions + table_center

    # 随机打乱位置顺序
    indices = th.randperm(final_positions.shape[0])
    final_positions = final_positions[indices][:available_positions]

    print(f"生成了 {final_positions.shape[0]} 个位置点")
    return final_positions, available_positions


def random_orientation(axis_aligned=True):
    """
    生成随机朝向四元数

    参数:
        axis_aligned (bool): 如果为True，只在Z轴上随机旋转，使物体保持直立
                          如果为False，生成完全随机的朝向

    返回:
        list: 形式为[x, y, z, w]的四元数
    """
    import torch as th

    if axis_aligned:
        # 只在Z轴上随机旋转 (0, 0, sin(θ/2), cos(θ/2))
        angle = random.uniform(0, 2 * math.pi)  # 随机角度
        return [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
    else:
        # 完全随机朝向，使用omnigibson提供的函数
        return random_quaternion(1).tolist()[0]


def generate_cluttered_objects(
    categories=None, num_objects=None, random_models=True, env=None, table_name="table", grid_size=0.1, **kwargs
):
    """
    生成多个物品的配置，使桌面变得杂乱

    参数:
        categories (list): 物品类别列表，如果为None，会随机选择常见物品类别
        num_objects (int或list): 物品数量，如果为None且categories为列表，则为categories的长度；
                               如果为list，则应与categories列表长度相同，表示每类物品的数量
        random_models (bool): 是否随机选择模型，否则选择第一个可用模型
        env (og.Environment): OmniGibson环境，用于获取桌子的实际尺寸
        table_name (str): 桌子的名称
        grid_size (float): 每个网格的固定大小
        **kwargs: 额外参数，如padding等

    返回:
        list: 包含多个物品配置的列表
    """

    # 如果没有提供类别，随机选择常见物品
    if categories is None:
        # 常见的小物品类别
        common_categories = [
            "apple",
            "banana",
            "bowl",
            "cup",
            "mug",
            "orange",
            "wineglass",
            "book",
            "pen",
            "fork",
            "knife",
            "spoon",
            "saltshaker",
            "peppershaker",
            "plate",
            "scissors",
            "cellphone",
            "tomato",
        ]
        # 获取所有可用的物品类别
        all_categories = get_all_object_categories()
        # 找出交集
        available_common = [cat for cat in common_categories if cat in all_categories]
        if len(available_common) == 0:
            # 如果常见物品都不存在，就随机选择
            categories = random.sample(all_categories, min(5, len(all_categories)))
        else:
            # 随机选择5个常见物品类别
            categories = random.sample(available_common, min(5, len(available_common)))

    # 处理物体数量配置
    if isinstance(num_objects, list):
        # 如果num_objects是列表，确保与categories长度一致
        if len(num_objects) != len(categories):
            print(f"警告: num_objects长度({len(num_objects)})与categories长度({len(categories)})不一致")
            if len(num_objects) < len(categories):
                # 如果num_objects较短，使用默认值1补齐
                num_objects = num_objects + [1] * (len(categories) - len(num_objects))
            else:
                # 如果num_objects较长，截断多余部分
                num_objects = num_objects[:len(categories)]
        
        # 计算期望的物体总数
        expected_total_objects = sum(num_objects)
    elif num_objects is not None:
        # 如果num_objects是单个数字，表示总数
        expected_total_objects = num_objects
        # 创建一个均匀分布的列表
        num_per_category = expected_total_objects // len(categories)
        remainder = expected_total_objects % len(categories)
        num_objects = [num_per_category] * len(categories)
        # 将余数分配给前几个类别
        for i in range(remainder):
            num_objects[i] += 1
    else:
        # 默认每个类别1个物体
        num_objects = [1] * len(categories)
        expected_total_objects = len(categories)

    print(f"期望的物体总数: {expected_total_objects}")
    print(f"分配给各类别的数量: {num_objects}")

    # 获取桌子的包围盒信息和可用位置数量
    positions = None
    available_positions = 0
    if env is not None:
        # 通过OmniGibson API获取桌子的实际包围盒
        bbox_center, bbox_extent, table_height, table_orientation = get_table_bbox(env, table_name)
        if bbox_center is not None:
            # 生成网格位置
            padding = kwargs.get("padding", 0.1)  # 获取padding参数，默认为0.1
            occupancy_rate = kwargs.get("occupancy_rate", 0.5)  # 获取occupancy_rate参数，默认为0.5

            # 仅根据网格数量和occupancy_rate确定生成的位置点数量
            # 生成包括所有网格点的位置，后续根据实际需要调整
            positions, available_positions = generate_grid_positions(
                bbox_center,
                bbox_extent,
                table_orientation,
                grid_size=grid_size,
                padding=padding,
                occupancy_rate=occupancy_rate,
            )

    # 如果无法获取桌子包围盒或生成网格位置失败，使用传统方法随机生成位置
    if positions is None:
        raise ValueError("无法获取桌子包围盒或生成网格位置失败")

    print(f"根据网格和占用率生成的可用位置数量: {available_positions}")

    # 根据available_positions与expected_total_objects的关系调整物体数量
    if available_positions < expected_total_objects:
        # 如果可用位置少于期望物体总数，按比例截断每个类别的数量
        print(f"警告: 可用位置({available_positions})少于期望物体总数({expected_total_objects})，将截断物体数量")
        scale_factor = available_positions / expected_total_objects
        new_num_objects = []
        total_allocated = 0
        
        # 首先分配整数部分
        for n in num_objects:
            allocated = int(n * scale_factor)
            new_num_objects.append(allocated)
            total_allocated += allocated
        
        # 分配剩余的位置
        remaining = available_positions - total_allocated
        i = 0
        while remaining > 0 and i < len(new_num_objects):
            new_num_objects[i] += 1
            remaining -= 1
            i += 1
        
        num_objects = new_num_objects
        expected_total_objects = available_positions
    elif available_positions > expected_total_objects and kwargs.get("auto_supplement", False):
        # 如果可用位置多于期望物体总数，并且启用了自动补充，可以增加物体数量
        # 注意：只有在配置中明确启用auto_supplement时才进行补充
        print(f"可用位置({available_positions})多于期望物体总数({expected_total_objects})，可以补充更多物体")
        extra_positions = available_positions - expected_total_objects
        
        # 按比例增加每个类别的数量
        if extra_positions > 0:
            original_total = sum(num_objects)
            for i in range(len(num_objects)):
                extra = int(extra_positions * (num_objects[i] / original_total))
                num_objects[i] += extra
                extra_positions -= extra
            
            # 分配剩余的位置
            i = 0
            while extra_positions > 0 and i < len(num_objects):
                num_objects[i] += 1
                extra_positions -= 1
                i += 1
            
            expected_total_objects = available_positions

    print(f"最终的物体分配数量: {num_objects}")
    print(f"最终的物体总数: {sum(num_objects)}")

    # 展开类别和数量列表
    expanded_categories = []
    for i, (category, count) in enumerate(zip(categories, num_objects)):
        expanded_categories.extend([category] * count)

    # 生成物品配置
    objects_cfg = []
    num_positions_used = 0

    for i, category in enumerate(expanded_categories):
        # 确保不超过可用位置数量
        if num_positions_used >= available_positions:
            break

        # 获取该类别的所有可用模型
        available_models = get_all_object_category_models(category)

        if len(available_models) == 0:
            print(f"警告: 类别 {category} 不存在或没有可用模型，跳过")
            continue

        # 选择模型
        if random_models:
            model = random.choice(available_models)
        else:
            model = available_models[0]

        # 设置位置
        if num_positions_used < positions.shape[0]:
            # 使用预生成的网格位置
            position = positions[num_positions_used].tolist()  # 转换为列表以适配DatasetObject
            num_positions_used += 1
        else:
            # 这种情况应该不会发生，因为我们已经限制了循环次数
            print(f"警告: 位置索引 {num_positions_used} 超出预生成位置范围 {positions.shape[0]}，跳过")
            continue

        # 创建物品配置
        obj_cfg = {
            "type": "DatasetObject",
            "name": f"{category}_{i+1}",
            "category": category,
            "model": model,
            "fixed_base": False,
            "position": position,
            "orientation": random_orientation(axis_aligned=kwargs.get("axis_aligned", False)),
        }

        objects_cfg.append(obj_cfg)

    print(f"已生成 {len(objects_cfg)} 个物品配置，使用位置数量: {num_positions_used}")

    return objects_cfg


def merge_cfg(cfg, objects_cfg):
    """
    将生成的物品配置添加到现有配置中

    参数:
        cfg (dict): 原始配置
        objects_cfg (list): 物品配置列表

    返回:
        dict: 更新后的配置
    """
    # 深拷贝原始配置
    new_cfg = copy.deepcopy(cfg)

    # 检查是否存在objects键并添加物品配置
    if "objects" not in new_cfg:
        new_cfg["objects"] = []

    new_cfg["objects"].extend(objects_cfg)

    return new_cfg


def add_dynamic_objects(custom_cfgs, env):
    """添加动态物体到环境并返回物体列表以便在重置时使用

    参数:
        custom_cfgs (dict): 自定义配置
        env (og.Environment): 环境实例

    返回:
        list: 添加的动态物体列表
    """
    import omnigibson as og
    from omnigibson.objects import DatasetObject

    dynamic_objects = []
    if custom_cfgs["random_table_objects"]:
        # 暂停模拟以安全地添加物体
        is_playing = og.sim.is_playing()
        if is_playing:
            og.sim.pause()

        try:
            objects_cfg = generate_cluttered_objects(**custom_cfgs["random_table_objects"], env=env, return_cfg=False)
            objects_instances = []

            # 创建物体实例
            for obj_cfg in objects_cfg:
                obj = DatasetObject(**obj_cfg)
                objects_instances.append(obj)

            # 批量添加所有对象
            for i, obj in enumerate(objects_instances):
                try:
                    env.scene.add_object(obj)
                    # 设置位置
                    obj.set_position_orientation(
                        position=objects_cfg[i].get("position", None),
                        orientation=objects_cfg[i].get("orientation", None),
                    )
                    dynamic_objects.append(obj)
                except Exception as e:
                    print(f"添加物体时出错: {e}")
        finally:
            # 恢复模拟状态
            if is_playing:
                og.sim.play()
                # 执行一次物理步骤让对象稳定
                og.sim.step_physics()

    print(f"成功添加了 {len(dynamic_objects)} 个动态物体")
    return dynamic_objects


def wrap_reset_for_dynamic_objects(env, custom_cfgs):
    """为环境添加自定义的reset方法，使其能够重置动态物体

    参数:
        env (og.Environment): 环境实例
        custom_cfgs (dict): 自定义配置字典

    返回:
        function: 修改后的reset方法
    """
    import omnigibson as og

    original_reset = env.reset

    def custom_reset(get_obs=True, **kwargs):
        # 确保模拟器在重置前处于播放状态
        was_paused = False
        if not og.sim.is_playing():
            print("重置前启动模拟状态...")
            was_paused = True
            og.sim.play()
            
        try:
            # 调用原始的reset方法
            result = original_reset(get_obs=False, **kwargs)

            # 移除所有动态添加的物体
            objects_to_remove = []
            if hasattr(env, "dynamic_objects") and env.dynamic_objects:
                # 收集所有要移除的对象
                for obj in env.dynamic_objects:
                    if obj is not None and hasattr(obj, "scene") and obj.scene is not None:
                        objects_to_remove.append(obj)

                # 清空动态对象列表
                env.dynamic_objects = []

                # 如果有对象需要移除，使用batch_remove_objects
                if objects_to_remove:
                    try:
                        # 暂停模拟以安全地移除物体
                        is_playing = og.sim.is_playing()
                        if is_playing:
                            og.sim.pause()

                        # 批量移除所有对象
                        og.sim.batch_remove_objects(objects_to_remove)

                        # 恢复模拟状态
                        if is_playing:
                            og.sim.play()
                    except Exception as e:
                        print(f"批量移除物体时出错: {e}")

            # 重新添加动态物体
            env.dynamic_objects = add_dynamic_objects(custom_cfgs, env)

            # 获取观察结果并返回
            if get_obs:
                og.sim.step()
                obs, _ = env.get_obs()
                return obs
            return result
        finally:
            # 如果之前是暂停状态，恢复到暂停状态
            if was_paused and og.sim.is_playing():
                og.sim.pause()

    # 替换环境的reset方法
    env.reset = custom_reset
    return custom_reset

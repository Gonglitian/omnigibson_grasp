#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OmniGibson物体包围盒信息获取脚本

该脚本用于在不加载物体的情况下获取OmniGibson中物体的包围盒信息。
"""

import omnigibson.lazy as lazy
import pxr
from omnigibson.utils.asset_utils import get_og_avg_category_specs
from omnigibson.macros import gm
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_object_bbox_before_loading(category, model):
    """
    在不加载物体的情况下获取物体包围盒信息

    参数:
        category (str): 物体类别
        model (str): 模型ID

    返回:
        tuple: 包围盒信息和平均尺寸信息
    """
    # 1. 尝试获取类别的平均尺寸规格
    avg_specs = get_og_avg_category_specs()
    category_avg_specs = avg_specs.get(category, None)

    # 2. 尝试直接从USD文件读取原始包围盒信息
    usd_path = os.path.join(
        gm.DATASET_PATH, "objects", category, model, "usd", f"{model}.encrypted.usd"
    )

    # 检查文件是否存在
    if not os.path.exists(usd_path):
        print(f"USD文件不存在: {usd_path}")
        return None, category_avg_specs

    try:
        # 打开USD文件
        stage = pxr.Usd.Stage.Open(usd_path)

        # 获取默认的prim
        default_prim = stage.GetDefaultPrim()

        # 尝试获取nativeBB属性
        if default_prim.HasAttribute("ig:nativeBB"):
            native_bbox = default_prim.GetAttribute("ig:nativeBB").Get()
            return native_bbox, category_avg_specs
    except Exception as e:
        print(f"读取USD文件时出错: {e}")

    return None, category_avg_specs


def calculate_volume_from_bbox(bbox):
    """
    根据包围盒计算体积

    参数:
        bbox: 包围盒信息，格式为[(min_x, min_y, min_z), (max_x, max_y, max_z)]

    返回:
        float: 体积
    """
    if bbox is None:
        return None

    min_point, max_point = bbox
    dimensions = [max_point[i] - min_point[i] for i in range(3)]
    return dimensions[0] * dimensions[1] * dimensions[2]


def visualize_bbox(bbox, ax=None):
    """
    可视化包围盒

    参数:
        bbox: 包围盒信息，格式为[(min_x, min_y, min_z), (max_x, max_y, max_z)]
        ax: matplotlib 3D轴，如果为None则创建新的

    返回:
        matplotlib.axes: 3D轴对象
    """
    if bbox is None:
        print("无包围盒信息可视化")
        return None

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    min_point, max_point = bbox

    # 计算8个顶点
    vertices = [
        [min_point[0], min_point[1], min_point[2]],
        [max_point[0], min_point[1], min_point[2]],
        [max_point[0], max_point[1], min_point[2]],
        [min_point[0], max_point[1], min_point[2]],
        [min_point[0], min_point[1], max_point[2]],
        [max_point[0], min_point[1], max_point[2]],
        [max_point[0], max_point[1], max_point[2]],
        [min_point[0], max_point[1], max_point[2]],
    ]

    # 定义边
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # 底面
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # 顶面
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # 连接顶面和底面的边
    ]

    # 绘制边
    for edge in edges:
        line = np.array([vertices[edge[0]], vertices[edge[1]]])
        ax.plot(line[:, 0], line[:, 1], line[:, 2], "b")

    # 设置轴范围和标签
    all_coords = np.array(vertices)
    ax.set_xlim([all_coords[:, 0].min() - 0.1, all_coords[:, 0].max() + 0.1])
    ax.set_ylim([all_coords[:, 1].min() - 0.1, all_coords[:, 1].max() + 0.1])
    ax.set_zlim([all_coords[:, 2].min() - 0.1, all_coords[:, 2].max() + 0.1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("物体包围盒可视化")

    # 保持纵横比一致
    ax.set_box_aspect([1, 1, 1])

    return ax


def get_available_categories():
    """
    获取OmniGibson中可用的物体类别

    返回:
        list: 类别列表
    """
    object_dir = os.path.join(gm.DATASET_PATH, "objects")
    if not os.path.exists(object_dir):
        print(f"物体目录不存在: {object_dir}")
        return []

    categories = [
        d for d in os.listdir(object_dir) if os.path.isdir(os.path.join(object_dir, d))
    ]
    return sorted(categories)


def get_models_for_category(category):
    """
    获取指定类别的所有模型

    参数:
        category (str): 物体类别

    返回:
        list: 模型ID列表
    """
    category_dir = os.path.join(gm.DATASET_PATH, "objects", category)
    if not os.path.exists(category_dir):
        print(f"类别目录不存在: {category_dir}")
        return []

    models = [
        d
        for d in os.listdir(category_dir)
        if os.path.isdir(os.path.join(category_dir, d))
    ]
    return sorted(models)


def display_object_info(category, model):
    """
    显示物体的包围盒和体积信息

    参数:
        category (str): 物体类别
        model (str): 模型ID
    """
    print(f"\n获取 {category}/{model} 的包围盒信息...")
    bbox, avg_specs = get_object_bbox_before_loading(category, model)

    if bbox is not None:
        print(f"\n包围盒信息:")
        print(f"  最小点: {bbox[0]}")
        print(f"  最大点: {bbox[1]}")

        dimensions = [bbox[1][i] - bbox[0][i] for i in range(3)]
        print(f"\n尺寸:")
        print(f"  宽度 (X): {dimensions[0]:.4f} 米")
        print(f"  深度 (Y): {dimensions[1]:.4f} 米")
        print(f"  高度 (Z): {dimensions[2]:.4f} 米")

        volume = calculate_volume_from_bbox(bbox)
        print(f"\n体积: {volume:.6f} 立方米")

        # 可视化
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, projection="3d")
        visualize_bbox(bbox, ax)
        plt.tight_layout()
        plt.show()
    else:
        print("无法获取包围盒信息，尝试使用类别平均规格...")

    if avg_specs is not None:
        print(f"\n类别平均规格:")
        for key, value in avg_specs.items():
            print(f"  {key}: {value}")
    else:
        print("无法获取类别平均规格信息")


def batch_get_volume_for_category(category, max_models=5):
    """
    批量获取指定类别的物体体积

    参数:
        category (str): 物体类别
        max_models (int): 最多处理的模型数量

    返回:
        dict: 模型ID到体积的映射
    """
    models = get_models_for_category(category)[:max_models]
    results = {}

    for model in models:
        print(f"处理 {category}/{model}...")
        bbox, _ = get_object_bbox_before_loading(category, model)
        volume = calculate_volume_from_bbox(bbox) if bbox is not None else None
        results[model] = {"bbox": bbox, "volume": volume}

    return results


def display_batch_results(category, max_models=5):
    """
    显示批量处理结果

    参数:
        category (str): 物体类别
        max_models (int): 最多处理的模型数量
    """
    print(f"批量获取 {category} 类别的体积信息 (最多 {max_models} 个模型)...")
    results = batch_get_volume_for_category(category, max_models)

    # 显示结果表格
    print("\n结果汇总:")
    print(f"{'模型ID':<20} {'体积 (立方米)':<15} {'有包围盒':<10}")
    print("-" * 45)

    valid_volumes = []
    for model, data in results.items():
        volume = data["volume"]
        has_bbox = data["bbox"] is not None
        volume_str = f"{volume:.6f}" if volume is not None else "N/A"
        print(f"{model:<20} {volume_str:<15} {'是' if has_bbox else '否':<10}")
        if volume is not None:
            valid_volumes.append(volume)

    if valid_volumes:
        avg_volume = sum(valid_volumes) / len(valid_volumes)
        print(f"\n平均体积: {avg_volume:.6f} 立方米")
        print(f"最小体积: {min(valid_volumes):.6f} 立方米")
        print(f"最大体积: {max(valid_volumes):.6f} 立方米")

        # 绘制体积分布图
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(valid_volumes)), valid_volumes)
        plt.grid(True, alpha=0.3)
        plt.xlabel("模型索引")
        plt.ylabel("体积 (立方米)")
        plt.title(f"{category} 类别的体积分布")
        plt.show()
    else:
        print("\n没有获取到有效的体积数据")


def compare_categories(categories, max_models=3):
    """
    比较多个类别的体积分布

    参数:
        categories (list): 要比较的类别列表
        max_models (int): 每个类别最多处理的模型数量
    """
    if not categories:
        print("请提供要比较的类别列表!")
        return

    if len(categories) > 5:
        print("为了清晰展示，最多选择5个类别进行比较")
        categories = categories[:5]

    print(
        f"比较 {', '.join(categories)} 类别的体积分布 (每类最多 {max_models} 个模型)..."
    )

    # 获取所有类别的体积数据
    category_volumes = {}
    for category in categories:
        results = batch_get_volume_for_category(category, max_models)
        volumes = [
            data["volume"] for data in results.values() if data["volume"] is not None
        ]
        if volumes:
            category_volumes[category] = volumes
        else:
            print(f"警告: 类别 {category} 没有有效的体积数据")

    if not category_volumes:
        print("没有获取到任何有效的体积数据!")
        return

    # 计算统计信息
    print("\n体积统计信息:")
    print(
        f"{'类别':<20} {'平均体积':<15} {'最小体积':<15} {'最大体积':<15} {'样本数':<10}"
    )
    print("-" * 75)

    for category, volumes in category_volumes.items():
        avg = sum(volumes) / len(volumes)
        min_vol = min(volumes)
        max_vol = max(volumes)
        print(
            f"{category:<20} {avg:.6f}{'':8} {min_vol:.6f}{'':8} {max_vol:.6f}{'':8} {len(volumes):<10}"
        )

    # 绘制箱线图比较
    plt.figure(figsize=(12, 8))
    plt.boxplot(
        [volumes for volumes in category_volumes.values()],
        labels=category_volumes.keys(),
    )
    plt.grid(True, alpha=0.3)
    plt.ylabel("体积 (立方米)")
    plt.title("不同类别物体的体积分布对比")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制条形图比较平均体积
    plt.figure(figsize=(12, 8))
    categories = list(category_volumes.keys())
    avg_volumes = [sum(volumes) / len(volumes) for volumes in category_volumes.values()]

    plt.bar(categories, avg_volumes)
    plt.grid(True, alpha=0.3)
    plt.ylabel("平均体积 (立方米)")
    plt.title("不同类别物体的平均体积对比")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_example_category():
    """
    展示常见物品类别的示例
    """
    categories = get_available_categories()
    common_categories = ["apple", "bowl", "mug", "book", "bottle"]

    # 检查这些类别是否可用
    available_common = [c for c in common_categories if c in categories]

    if available_common:
        print(f"可用的常见物品类别: {available_common}")

        # 选择第一个可用类别作为示例
        example_category = available_common[0]
        models = get_models_for_category(example_category)[:3]  # 取前3个模型

        if models:
            print(f"\n{example_category} 类别的前3个模型:")
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")

            # 展示第一个模型的包围盒信息
            display_object_info(example_category, models[0])
        else:
            print(f"没有找到 {example_category} 类别的模型")
    else:
        print("没有找到常见物品类别，请确保OmniGibson数据集路径正确")


def display_menu():
    """
    显示主菜单
    """
    print("\n" + "=" * 50)
    print("物体包围盒信息获取工具")
    print("=" * 50)
    print("1. 查看可用物体类别")
    print("2. 查看单个物体信息")
    print("3. 批量获取类别物体体积")
    print("4. 比较多个类别体积分布")
    print("5. 查看常见物品示例")
    print("0. 退出")
    print("=" * 50)

    choice = input("请输入选项编号: ")
    return choice


def main():
    """
    主函数
    """
    print("欢迎使用物体包围盒信息获取工具！")
    print("该工具可以在不加载物体的情况下获取OmniGibson中物体的包围盒和体积信息。")

    # 获取可用类别
    categories = get_available_categories()
    print(f"发现 {len(categories)} 个物体类别")
    if categories:
        print(f"前5个类别: {categories[:5]}")
    else:
        print("未找到可用的物体类别，请确保OmniGibson数据集路径正确")
        return

    while True:
        choice = display_menu()

        if choice == "0":
            print("谢谢使用，再见！")
            break

        elif choice == "1":
            print("\n可用的物体类别:")
            for i, category in enumerate(categories):
                print(f"{i+1}. {category}")

            input("\n按Enter键继续...")

        elif choice == "2":
            print("\n可用的物体类别:")
            for i, category in enumerate(categories):
                print(f"{i+1}. {category}")

            cat_idx = int(input("\n请选择类别编号: ")) - 1
            if 0 <= cat_idx < len(categories):
                selected_category = categories[cat_idx]
                models = get_models_for_category(selected_category)

                if models:
                    print(f"\n{selected_category} 类别的可用模型:")
                    for i, model in enumerate(models[:20]):  # 只显示前20个
                        print(f"{i+1}. {model}")

                    if len(models) > 20:
                        print(f"...还有 {len(models)-20} 个模型未显示")

                    model_idx = int(input("\n请选择模型编号: ")) - 1
                    if 0 <= model_idx < len(models):
                        selected_model = models[model_idx]
                        display_object_info(selected_category, selected_model)
                    else:
                        print("无效的模型编号!")
                else:
                    print(f"没有找到 {selected_category} 类别的模型")
            else:
                print("无效的类别编号!")

        elif choice == "3":
            print("\n可用的物体类别:")
            for i, category in enumerate(categories):
                print(f"{i+1}. {category}")

            cat_idx = int(input("\n请选择类别编号: ")) - 1
            if 0 <= cat_idx < len(categories):
                selected_category = categories[cat_idx]
                max_models = int(input("请输入最大处理模型数量: "))
                display_batch_results(selected_category, max_models)
            else:
                print("无效的类别编号!")

        elif choice == "4":
            print("\n可用的物体类别:")
            for i, category in enumerate(categories):
                print(f"{i+1}. {category}")

            cat_indices = input(
                "\n请选择要比较的类别编号（用逗号分隔，如1,3,5）: "
            ).split(",")
            selected_categories = []

            for idx in cat_indices:
                try:
                    cat_idx = int(idx.strip()) - 1
                    if 0 <= cat_idx < len(categories):
                        selected_categories.append(categories[cat_idx])
                    else:
                        print(f"忽略无效的类别编号: {idx}")
                except ValueError:
                    print(f"忽略无效输入: {idx}")

            if selected_categories:
                max_models = int(input("请输入每类最大处理模型数量: "))
                compare_categories(selected_categories, max_models)
            else:
                print("未选择有效的类别!")

        elif choice == "5":
            show_example_category()

        else:
            print("无效的选项，请重新输入！")


if __name__ == "__main__":
    main()

import os
import sys
import io
from pathlib import Path
from utils.project_tree import generate_tree

def concat_files(output_path, ignore_dirs=None, ignore_extensions=None, include_extensions=None):
    """将项目中的所有文件整合到一个txt文件中"""
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.history', 'build', 'dist', 'venv', 'env', '.venv']
    
    if ignore_extensions is None:
        ignore_extensions = ['.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.mp3', '.mp4', '.zip', '.tar', '.gz', '.rar']
    
    # 项目根目录
    root_dir = os.path.abspath('.')
    
    # 使用utils.project_tree中的generate_tree函数生成项目树结构
    # 创建一个临时文件来捕获输出
    tree_output_file = "temp_tree_output.txt"
    generate_tree(
        root_dir, 
        output_file=tree_output_file,
        show_files=True,
        ignore_dirs=ignore_dirs,
        ignore_patterns=['*.pyc', '*.pyo', '*.pyd'],
        max_depth=None,
        max_width=120,
        use_colors=False,  # 不使用颜色，因为要写入文本文件
        use_icons=True     # 保留图标
    )
    
    # 读取临时文件内容
    with open(tree_output_file, 'r', encoding='utf-8') as tree_file:
        tree_content = tree_file.read()
    
    # 删除临时文件
    os.remove(tree_output_file)
    
    # 添加标题
    tree_structure = "项目结构：\n" + tree_content
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        # 写入项目树结构
        output_file.write(tree_structure)
        output_file.write("\n\n" + "=" * 80 + "\n\n")
        
        # 递归遍历并写入文件内容
        for root, dirs, files in os.walk(root_dir):
            # 忽略指定目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            
            for file in sorted(files):
                file_path = os.path.join(root, file)
                
                # 跳过输出文件本身
                if os.path.abspath(file_path) == os.path.abspath(output_path):
                    continue
                
                # 跳过临时树输出文件
                if os.path.abspath(file_path) == os.path.abspath(tree_output_file):
                    continue
                
                # 检查文件扩展名
                _, ext = os.path.splitext(file)
                
                # 如果指定了要包含的扩展名，则只处理这些扩展名的文件
                if include_extensions and ext.lower() not in include_extensions:
                    continue
                
                # 如果文件扩展名在忽略列表中，则跳过
                if ext.lower() in ignore_extensions:
                    continue
                    
                # 计算相对路径
                rel_path = os.path.relpath(file_path, root_dir)
                
                try:
                    # 检查文件是否为文本文件
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # 写入文件路径和分隔符
                    output_file.write(f"文件: {rel_path}\n")
                    output_file.write("-" * 80 + "\n\n")
                    
                    # 写入文件内容
                    output_file.write(content)
                    output_file.write("\n\n" + "=" * 80 + "\n\n")
                except Exception as e:
                    output_file.write(f"文件: {rel_path} (无法读取: {str(e)})\n")
                    output_file.write("-" * 80 + "\n\n")
                    output_file.write("\n\n" + "=" * 80 + "\n\n")

if __name__ == "__main__":
    # 定义要忽略的目录
    ignore_dirs = ['.git', '__pycache__', '.history', 'build', 'dist', 'venv', 'env', '.venv', 'omnigibson', "ompl_lib"]
    
    # 定义输出文件路径
    output_file = "project_concat.txt"
    
    # 定义要包含的文件后缀
    include_extensions = ['.py', '.md']
    
    # 命令行参数处理
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    # 如果命令行参数提供了要忽略的目录，则添加
    ignore_flag_index = -1
    include_flag_index = -1
    
    for i, arg in enumerate(sys.argv):
        if arg == "--ignore-dirs" and i + 1 < len(sys.argv):
            ignore_flag_index = i
            break
    
    for i, arg in enumerate(sys.argv):
        if arg == "--include-ext" and i + 1 < len(sys.argv):
            include_flag_index = i
            break
    
    # 处理要忽略的目录
    if ignore_flag_index > 0 and ignore_flag_index + 1 < len(sys.argv):
        additional_ignore_dirs = sys.argv[ignore_flag_index + 1].split(',')
        ignore_dirs.extend(additional_ignore_dirs)
    
    # 处理要包含的文件后缀
    if include_flag_index > 0 and include_flag_index + 1 < len(sys.argv):
        include_extensions = [f".{ext.lstrip('.')}" for ext in sys.argv[include_flag_index + 1].split(',')]
    
    print(f"开始整合项目文件到 {output_file}")
    print(f"忽略目录: {', '.join(ignore_dirs)}")
    print(f"包含文件类型: {', '.join(include_extensions)}")
    
    concat_files(output_file, ignore_dirs, None, include_extensions)
    
    print(f"项目文件整合完成，输出文件: {output_file}") 
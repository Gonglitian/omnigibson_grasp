import os
import sys
import io
import argparse
from pathlib import Path
import fnmatch

# 尝试导入colorama，如果失败则创建替代品
try:
    import colorama
    from colorama import Fore, Style
    has_colorama = True
    # 初始化colorama
    colorama.init()
except ImportError:
    has_colorama = False
    # 创建替代品
    class DummyColor:
        def __init__(self):
            self.BLACK = ''
            self.RED = ''
            self.GREEN = ''
            self.YELLOW = ''
            self.BLUE = ''
            self.MAGENTA = ''
            self.CYAN = ''
            self.WHITE = ''
            self.LIGHTBLACK_EX = ''
            self.LIGHTRED_EX = ''
            self.LIGHTGREEN_EX = ''
            self.LIGHTYELLOW_EX = ''
            self.LIGHTBLUE_EX = ''
            self.LIGHTMAGENTA_EX = ''
            self.LIGHTCYAN_EX = ''
            self.LIGHTWHITE_EX = ''
    
    class DummyStyle:
        RESET_ALL = ''
    
    Fore = DummyColor()
    Style = DummyStyle()
    print("警告: 未安装colorama库，颜色功能将被禁用")

def generate_tree(
    root_path, 
    output_file=None,
    show_files=True,
    ignore_dirs=None,
    ignore_patterns=None,
    max_depth=None,
    max_width=120,
    use_colors=True,
    use_icons=True
):
    """
    生成项目树结构并输出到控制台或文件
    
    参数:
        root_path (str): 根目录路径
        output_file (str): 输出文件路径，如果为 None 则输出到控制台
        show_files (bool): 是否显示文件
        ignore_dirs (list): 要忽略的目录列表
        ignore_patterns (list): 要忽略的文件模式列表
        max_depth (int): 最大目录深度，None 表示无限制
        max_width (int): 最大行宽
        use_colors (bool): 是否使用颜色
        use_icons (bool): 是否使用图标
    """
    # 默认忽略的目录和文件
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.vscode', '.idea', 'node_modules', 'venv', 'env', '.env']
    
    if ignore_patterns is None:
        ignore_patterns = ['*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.class', '*.DS_Store']
    
    # 图标映射
    icons = {
        'folder': '📂 ' if use_icons else '',
        'file': '📄 ' if use_icons else '',
        'python': '🐍 ' if use_icons else '',
        'markdown': '📝 ' if use_icons else '',
        'html': '🌐 ' if use_icons else '',
        'css': '🎨 ' if use_icons else '',
        'js': '⚡ ' if use_icons else '',
        'json': '📊 ' if use_icons else '',
        'image': '🖼️ ' if use_icons else '',
        'audio': '🔊 ' if use_icons else '',
        'video': '🎬 ' if use_icons else '',
        'archive': '📦 ' if use_icons else '',
        'pdf': '📑 ' if use_icons else '',
        'executable': '⚙️ ' if use_icons else '',
        'unknown': '❓ ' if use_icons else ''
    }
    
    # 文件扩展名到图标的映射
    extension_to_icon = {
        '.py': 'python',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.js': 'js',
        '.json': 'json',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.gif': 'image',
        '.bmp': 'image',
        '.svg': 'image',
        '.mp3': 'audio',
        '.wav': 'audio',
        '.flac': 'audio',
        '.mp4': 'video',
        '.avi': 'video',
        '.mov': 'video',
        '.mkv': 'video',
        '.zip': 'archive',
        '.tar': 'archive',
        '.gz': 'archive',
        '.rar': 'archive',
        '.pdf': 'pdf',
        '.exe': 'executable',
        '.bat': 'executable',
        '.sh': 'executable',
    }
    
    # 颜色映射
    colors = {
        'folder': Fore.BLUE if use_colors else '',
        'file': Fore.WHITE if use_colors else '',
        'python': Fore.GREEN if use_colors else '',
        'markdown': Fore.CYAN if use_colors else '',
        'html': Fore.MAGENTA if use_colors else '',
        'css': Fore.LIGHTMAGENTA_EX if use_colors else '',
        'js': Fore.YELLOW if use_colors else '',
        'json': Fore.LIGHTCYAN_EX if use_colors else '',
        'image': Fore.LIGHTGREEN_EX if use_colors else '',
        'audio': Fore.LIGHTRED_EX if use_colors else '',
        'video': Fore.LIGHTRED_EX if use_colors else '',
        'archive': Fore.RED if use_colors else '',
        'pdf': Fore.RED if use_colors else '',
        'executable': Fore.RED if use_colors else '',
        'unknown': Fore.LIGHTBLACK_EX if use_colors else ''
    }
    
    # 打开输出文件或使用标准输出
    output = open(output_file, 'w', encoding='utf-8') if output_file else sys.stdout
    
    try:
        # 输出根目录名称
        root_name = os.path.basename(os.path.abspath(root_path))
        output.write(f"{colors['folder'] if use_colors else ''}{icons['folder']}{root_name}{Style.RESET_ALL if use_colors else ''}\n")
        
        # 递归生成树
        _generate_tree_recursive(
            Path(root_path), 
            output, 
            "", 
            True, 
            show_files,
            ignore_dirs,
            ignore_patterns,
            max_depth,
            1,
            max_width,
            icons,
            extension_to_icon,
            colors,
            use_colors
        )
    finally:
        # 关闭文件（如果使用文件输出）
        if output_file:
            output.close()

def _generate_tree_recursive(
    path, 
    output, 
    prefix, 
    is_last, 
    show_files,
    ignore_dirs,
    ignore_patterns,
    max_depth,
    current_depth,
    max_width,
    icons,
    extension_to_icon,
    colors,
    use_colors
):
    """递归生成树的辅助函数"""
    # 检查最大深度
    if max_depth is not None and current_depth > max_depth:
        return
    
    # 如果当前目录应该被忽略，则跳过
    if path.name in ignore_dirs or path.name.startswith('.'):
        return
    
    # 获取目录内容并排序（先文件夹后文件）
    try:
        items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        # 处理权限错误
        output.write(f"{prefix}{'└── ' if is_last else '├── '}[权限不足]\n")
        return
    
    # 过滤掉需要忽略的项目
    filtered_items = []
    for item in items:
        # 跳过被忽略的目录
        if item.is_dir() and (item.name in ignore_dirs or item.name.startswith('.')):
            continue
        
        # 跳过被忽略的文件模式
        if item.is_file() and any(fnmatch.fnmatch(item.name, pattern) for pattern in ignore_patterns):
            continue
        
        # 如果不显示文件并且当前项是文件，则跳过
        if not show_files and item.is_file():
            continue
        
        filtered_items.append(item)
    
    # 遍历过滤后的项目
    for i, item in enumerate(filtered_items):
        is_item_last = i == len(filtered_items) - 1
        
        # 确定前缀
        item_prefix = prefix + ('└── ' if is_last else '├── ')
        next_prefix = prefix + ('    ' if is_last else '│   ')
        
        # 确定图标和颜色
        if item.is_dir():
            icon_type = 'folder'
        else:
            ext = item.suffix.lower()
            icon_type = extension_to_icon.get(ext, 'unknown')
        
        icon = icons.get(icon_type, icons['unknown'])
        color = colors.get(icon_type, colors['unknown'])
        
        # 构建显示名称
        name = item.name
        if len(item_prefix) + len(name) > max_width:
            # 如果名称太长，进行截断
            half_width = (max_width - len(item_prefix) - 3) // 2  # 3是省略号的长度
            name = name[:half_width] + '...' + name[-half_width:]
        
        # 输出当前项
        output.write(f"{item_prefix}{color if use_colors else ''}{icon}{name}{Style.RESET_ALL if use_colors else ''}\n")
        
        # 如果是目录，递归处理
        if item.is_dir():
            _generate_tree_recursive(
                item, 
                output, 
                next_prefix, 
                is_item_last, 
                show_files,
                ignore_dirs,
                ignore_patterns,
                max_depth,
                current_depth + 1,
                max_width,
                icons,
                extension_to_icon,
                colors,
                use_colors
            )

def is_rich_available():
    """检查是否可以使用rich库"""
    try:
        import rich
        return True
    except ImportError:
        return False

def generate_tree_with_rich(
    root_path, 
    output_file=None,
    show_files=True,
    ignore_dirs=None,
    ignore_patterns=None,
    max_depth=None
):
    """使用rich库生成项目树（更漂亮的输出）"""
    try:
        from rich.tree import Tree
        from rich.console import Console
        from rich import print as rich_print
        
        # 默认忽略的目录和文件
        if ignore_dirs is None:
            ignore_dirs = ['.git', '__pycache__', '.vscode', '.idea', 'node_modules', 'venv', 'env', '.env']
        
        if ignore_patterns is None:
            ignore_patterns = ['*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.class', '*.DS_Store']
        
        console = Console(file=open(output_file, 'w', encoding='utf-8') if output_file else None)
        
        # 创建根节点
        root_name = os.path.basename(os.path.abspath(root_path))
        tree = Tree(f"[bold blue]:open_file_folder: {root_name}[/bold blue]")
        
        def add_to_tree(path, tree_node, depth=0):
            """递归添加节点到树"""
            # 检查最大深度
            if max_depth is not None and depth >= max_depth:
                return
            
            # 获取目录内容并排序
            try:
                items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                tree_node.add("[red][权限不足][/red]")
                return
            
            # 处理每个项目
            for item in items:
                # 跳过忽略的目录
                if item.is_dir() and (item.name in ignore_dirs or item.name.startswith('.')):
                    continue
                
                # 跳过忽略的文件模式
                if item.is_file() and any(fnmatch.fnmatch(item.name, pattern) for pattern in ignore_patterns):
                    continue
                
                # 如果不显示文件并且当前项是文件，则跳过
                if not show_files and item.is_file():
                    continue
                
                # 根据文件类型设置样式
                if item.is_dir():
                    style = "[bold blue]"
                    icon = ":open_file_folder:"
                elif item.suffix.lower() in ['.py']:
                    style = "[green]"
                    icon = ":snake:"
                elif item.suffix.lower() in ['.md', '.markdown']:
                    style = "[cyan]"
                    icon = ":memo:"
                elif item.suffix.lower() in ['.html', '.htm']:
                    style = "[magenta]"
                    icon = ":globe_with_meridians:"
                elif item.suffix.lower() in ['.css']:
                    style = "[purple]"
                    icon = ":art:"
                elif item.suffix.lower() in ['.js']:
                    style = "[yellow]"
                    icon = ":zap:"
                elif item.suffix.lower() in ['.json']:
                    style = "[cyan]"
                    icon = ":bar_chart:"
                elif item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']:
                    style = "[green]"
                    icon = ":frame_with_picture:"
                else:
                    style = "[white]"
                    icon = ":page_facing_up:"
                
                # 添加到树
                node_text = f"{style}{icon} {item.name}[/{style.strip('[]')}]"
                node = tree_node.add(node_text)
                
                # 如果是目录，继续递归
                if item.is_dir():
                    add_to_tree(item, node, depth + 1)
        
        # 开始递归
        add_to_tree(Path(root_path), tree)
        
        # 显示树
        console.print(tree)
        
        # 关闭文件（如果使用文件输出）
        if output_file and console.file:
            console.file.close()
            
    except ImportError as e:
        print(f"无法使用rich库: {e}")
        # 回退到基本实现
        generate_tree(
            root_path, 
            output_file,
            show_files,
            ignore_dirs,
            ignore_patterns,
            max_depth
        )

def concat_files(
    root_path=None, 
    output_path=None, 
    ignore_dirs=None, 
    ignore_extensions=None, 
    include_extensions=None,
    add_tree=True
):
    """
    将项目中的文件整合到一个txt文件中
    
    参数:
        root_path (str): 项目根目录路径，默认为当前目录
        output_path (str): 输出文件路径
        ignore_dirs (list): 要忽略的目录列表
        ignore_extensions (list): 要忽略的文件扩展名列表
        include_extensions (list): 要包含的文件扩展名列表，如果指定，则只包含这些扩展名的文件
        add_tree (bool): 是否在文件开头添加项目树结构
    """
    if root_path is None:
        root_path = os.path.abspath('.')
    
    if output_path is None:
        output_path = "project_concat.txt"
    
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.history', 'build', 'dist', 'venv', 'env', '.venv']
    
    if ignore_extensions is None:
        ignore_extensions = ['.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.mp3', '.mp4', '.zip', '.tar', '.gz', '.rar']
    
    # 创建临时文件用于存储项目树
    tree_structure = ""
    
    if add_tree:
        # 生成项目树结构
        tree_output_file = "temp_tree_output.txt"
        generate_tree(
            root_path, 
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
        # 写入项目树结构（如果需要）
        if add_tree:
            output_file.write(tree_structure)
            output_file.write("\n\n" + "=" * 80 + "\n\n")
        
        # 递归遍历并写入文件内容
        for root, dirs, files in os.walk(root_path):
            # 忽略指定目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            
            for file in sorted(files):
                file_path = os.path.join(root, file)
                
                # 跳过输出文件本身
                if os.path.abspath(file_path) == os.path.abspath(output_path):
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
                rel_path = os.path.relpath(file_path, root_path)
                
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
    
    return output_path

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='项目工具 - 生成目录树和整合项目文件')
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest='command', help='要执行的操作')
    
    # 生成树的命令
    tree_parser = subparsers.add_parser('tree', help='生成项目目录树')
    tree_parser.add_argument('path', nargs='?', default='.', help='项目根目录路径（默认为当前目录）')
    tree_parser.add_argument('-o', '--output', help='输出文件路径（默认为标准输出）')
    tree_parser.add_argument('-f', '--files', action='store_true', default=True, help='显示文件（默认开启）')
    tree_parser.add_argument('--no-files', dest='files', action='store_false', help='不显示文件')
    tree_parser.add_argument('-i', '--ignore', nargs='+', default=[], help='额外要忽略的目录或文件')
    tree_parser.add_argument('-d', '--depth', type=int, help='最大目录深度')
    tree_parser.add_argument('--no-colors', dest='colors', action='store_false', default=True, help='关闭颜色输出')
    tree_parser.add_argument('--no-icons', dest='icons', action='store_false', default=True, help='关闭图标输出')
    tree_parser.add_argument('--rich', action='store_true', default=False, help='使用rich库生成更漂亮的输出')
    
    # 整合文件的命令
    concat_parser = subparsers.add_parser('concat', help='整合项目文件到单个文本文件')
    concat_parser.add_argument('path', nargs='?', default='.', help='项目根目录路径（默认为当前目录）')
    concat_parser.add_argument('-o', '--output', default='project_concat.txt', help='输出文件路径（默认为project_concat.txt）')
    concat_parser.add_argument('-i', '--ignore-dirs', nargs='+', default=[], help='额外要忽略的目录')
    concat_parser.add_argument('-e', '--include-ext', nargs='+', default=['.py', '.md'], help='要包含的文件扩展名（默认为.py和.md）')
    concat_parser.add_argument('--no-tree', dest='add_tree', action='store_false', default=True, help='不在文件开头添加项目树结构')
    
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 处理生成树的命令
    if args.command == 'tree':
        # 合并忽略列表
        ignore_dirs = ['.git', '__pycache__', '.vscode', '.idea', 'node_modules', 'venv', 'env', '.env']
        ignore_dirs.extend([d for d in args.ignore if not d.startswith('*')])
        
        ignore_patterns = ['*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.class', '*.DS_Store']
        ignore_patterns.extend([p for p in args.ignore if p.startswith('*')])
        
        # 根据参数选择实现
        if args.rich and is_rich_available():
            generate_tree_with_rich(
                args.path,
                args.output,
                args.files,
                ignore_dirs,
                ignore_patterns,
                args.depth
            )
        else:
            generate_tree(
                args.path,
                args.output,
                args.files,
                ignore_dirs,
                ignore_patterns,
                args.depth,
                120,
                args.colors,
                args.icons
            )
        
        # 如果输出到文件，打印成功消息
        if args.output:
            print(f"项目树已保存到 {args.output}")
    
    # 处理整合文件的命令
    elif args.command == 'concat':
        # 合并忽略目录列表
        ignore_dirs = ['.git', '__pycache__', '.history', 'build', 'dist', 'venv', 'env', '.venv']
        ignore_dirs.extend(args.ignore_dirs)
        
        # 格式化文件扩展名
        include_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in args.include_ext]
        
        print(f"开始整合项目文件到 {args.output}")
        print(f"忽略目录: {', '.join(ignore_dirs)}")
        print(f"包含文件类型: {', '.join(include_extensions)}")
        
        output_path = concat_files(
            args.path,
            args.output,
            ignore_dirs,
            None,  # 使用默认的忽略扩展名
            include_extensions,
            args.add_tree
        )
        
        print(f"项目文件整合完成，输出文件: {output_path}")

# 提供后向兼容性的函数
def generate_concat_file(
    output_file="project_concat.txt", 
    ignore_dirs=None, 
    include_extensions=None
):
    """
    为了向后兼容而提供的函数，相当于旧的concat_project.py的功能
    
    参数:
        output_file (str): 输出文件路径
        ignore_dirs (list): 要忽略的目录列表
        include_extensions (list): 要包含的文件扩展名列表
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.history', 'build', 'dist', 'venv', 'env', '.venv', 'omnigibson', "ompl_lib"]
    
    if include_extensions is None:
        include_extensions = ['.py', '.md']
    
    print(f"开始整合项目文件到 {output_file}")
    print(f"忽略目录: {', '.join(ignore_dirs)}")
    print(f"包含文件类型: {', '.join(include_extensions)}")
    
    concat_files(
        None,  # 使用当前目录
        output_file,
        ignore_dirs,
        None,  # 使用默认的忽略扩展名
        include_extensions,
        True   # 添加项目树
    )
    
    print(f"项目文件整合完成，输出文件: {output_file}")

if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1 and sys.argv[1] not in ['tree', 'concat', '-h', '--help']:
        # 如果第一个参数不是子命令或帮助，则假定为旧的concat_project.py调用方式
        
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
        
        generate_concat_file(output_file, ignore_dirs, include_extensions)
    else:
        # 否则使用新的命令行接口
        main() 
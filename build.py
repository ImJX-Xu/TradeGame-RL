"""
打包脚本 - 使用 PyInstaller 打包游戏

使用方法：
    python build.py

打包后的文件：
    dist/风物千程.exe (Windows)
    或 dist/风物千程 (Linux/Mac)

存档位置：
    打包后，存档会保存在可执行文件同级目录下的 saves/ 文件夹
"""
import os
import sys
import shutil
from pathlib import Path

def main():
    try:
        import PyInstaller.__main__
    except ImportError:
        print("错误：未安装 PyInstaller")
        print("请运行：pip install pyinstaller")
        sys.exit(1)
    
    # 项目根目录
    project_root = Path(__file__).parent
    trade_game_dir = project_root / "trade_game"
    
    # 检查必需文件
    required_files = [
        trade_game_dir / "cities.csv",
        trade_game_dir / "products.csv",
        trade_game_dir / "routes.csv",
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("错误：缺少必需文件：")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)
    
    print("开始打包...")
    print(f"项目目录：{project_root}")
    
    # PyInstaller 参数
    args = [
        "start_game.py",
        "--name=风物千程",
        "--onefile",  # 单文件模式
        "--windowed",  # 无控制台窗口（Windows）
        "--clean",  # 清理临时文件
        "--noconfirm",  # 覆盖输出目录
        
        # 包含 CSV 文件（Windows 用 ;，Linux/Mac 用 :）
        f"--add-data={str(trade_game_dir / 'cities.csv')};trade_game",
        f"--add-data={str(trade_game_dir / 'products.csv')};trade_game",
        f"--add-data={str(trade_game_dir / 'routes.csv')};trade_game",
    ]
    
    # 如果存在 assets 目录，包含它
    assets_dir = trade_game_dir / "assets"
    if assets_dir.exists():
        # PyInstaller 目录格式：源目录;目标目录（Windows）或 源目录:目标目录（Linux/Mac）
        import os
        sep = ";" if os.name == "nt" else ":"
        args.append(f"--add-data={str(assets_dir)}{sep}trade_game/assets")
        print("✓ 包含 assets 目录")
    
    # 隐藏导入（确保所有模块都被包含）
    args.extend([
        "--hidden-import=arcade",
        "--hidden-import=arcade.key",
        "--hidden-import=arcade.color",
    ])
    
    # 图标（如果有）
    icon_path = project_root / "icon.ico"
    if icon_path.exists():
        args.append(f"--icon={icon_path}")
        print("✓ 使用自定义图标")
    
    print("\n执行 PyInstaller...")
    print(f"命令：pyinstaller {' '.join(args)}\n")
    
    # 执行打包
    PyInstaller.__main__.run(args)
    
    print("\n✓ 打包完成！")
    print(f"可执行文件位置：{project_root / 'dist' / '风物千程.exe'}")
    print("\n注意事项：")
    print("1. 存档会保存在可执行文件同级目录下的 saves/ 文件夹")
    print("2. 首次运行会自动创建 saves/ 目录")
    print("3. 如果遇到问题，请检查 dist/ 目录下的文件")

if __name__ == "__main__":
    main()

"""
打包前检查脚本

检查所有必需文件是否存在，确保打包前准备就绪。

使用方法：
    python check_build.py
"""
from pathlib import Path
import sys

def check_files():
    """检查打包所需的所有文件"""
    project_root = Path(__file__).parent
    trade_game_dir = project_root / "trade_game"
    
    print("=" * 60)
    print("打包前文件检查")
    print("=" * 60)
    print(f"项目根目录：{project_root}\n")
    
    # 必需文件列表
    required_files = {
        "启动文件": [
            project_root / "start_game.py",
        ],
        "CSV 数据文件": [
            trade_game_dir / "cities.csv",
            trade_game_dir / "products.csv",
            trade_game_dir / "routes.csv",
        ],
        "核心 Python 模块": [
            trade_game_dir / "__init__.py",
            trade_game_dir / "arcade_app.py",
            trade_game_dir / "data.py",
            trade_game_dir / "save_load.py",
            trade_game_dir / "game_config.py",
        ],
    }
    
    all_ok = True
    
    for category, files in required_files.items():
        print(f"【{category}】")
        for file_path in files:
            exists = file_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file_path.name}")
            if not exists:
                print(f"    路径：{file_path}")
                all_ok = False
        print()
    
    # 可选文件
    optional_files = {
        "资源文件": [
            trade_game_dir / "assets" / "china_map.png",
        ],
        "文档文件": [
            project_root / "README.md",
        ],
    }
    
    print("【可选文件】")
    for category, files in optional_files.items():
        for file_path in files:
            exists = file_path.exists()
            status = "✓" if exists else "○"
            print(f"  {status} {file_path.name}")
    print()
    
    # 检查依赖
    print("【依赖检查】")
    try:
        import arcade
        print(f"  ✓ arcade (版本: {arcade.__version__})")
    except ImportError:
        print("  ✗ arcade 未安装")
        all_ok = False
    
    try:
        import PyInstaller
        print(f"  ✓ PyInstaller (版本: {PyInstaller.__version__})")
    except ImportError:
        print("  ○ PyInstaller 未安装（打包时需要）")
    
    print()
    
    # 总结
    print("=" * 60)
    if all_ok:
        print("✓ 所有必需文件检查通过！可以开始打包。")
        print("\n下一步：运行 python build.py 进行打包")
    else:
        print("✗ 发现缺失文件，请先补充后再打包。")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    check_files()

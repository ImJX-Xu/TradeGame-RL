# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['start_game.py'],
    pathex=[],
    binaries=[],
    datas=[('D:\\Code\\TRADE\\trade_game\\cities.csv', 'trade_game'), ('D:\\Code\\TRADE\\trade_game\\products.csv', 'trade_game'), ('D:\\Code\\TRADE\\trade_game\\routes.csv', 'trade_game'), ('D:\\Code\\TRADE\\trade_game\\assets', 'trade_game/assets')],
    hiddenimports=['arcade', 'arcade.key', 'arcade.color'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='风物千程',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

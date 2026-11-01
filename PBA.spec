# -*- mode: python ; coding: utf-8 -*-
# PBA (Puyo Beginner Assistance) の PyInstaller 設定。
# ビルド: tools/build_exe.py を実行する(dist/PBA/ に PBA.exe と必要ファイルを揃える)。

block_cipher = None

a = Analysis(
    ["pba_launcher.py"],
    pathex=["."],
    binaries=[],
    # 認識の見本画像(src/vision/templates、Path(__file__) から読む)と、定跡・配ぷよの表(resource_root() の assets/)は
    # 読み取り専用データとして同梱する
    datas=[
        ("src/vision/templates/*.png", "src/vision/templates"),
        ("assets/opening_6moves.json", "assets"),
        ("assets/haipuyo.txt.gz", "assets"),
    ],
    # プラグイン(exe の隣の plugins/)が使う本体のモジュール。exe の中から import できるように入れておく
    hiddenimports=["src.plugin_api"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "PIL", "vgamepad"],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PBA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # コンソールを出さない(start.bat の pythonw と同じ)
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="PBA",
)

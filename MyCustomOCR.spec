
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
onnxruntime_datas = collect_data_files("onnxruntime", include_py_files=False)

a = Analysis(
    ["gui/app.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("data", "data"),
        ("weights", "weights"),
        ("checkpoints", "checkpoints"),
        ("src", "src"),
        ("gui", "gui"),
    ] + onnxruntime_datas,
    hiddenimports=[
        "onnxruntime", "onnxruntime.capi", "onnxruntime.capi._pybind_state",
        "cv2", "albumentations", "pandas",
        "torch", "torchvision",
        "PyQt6", "PyQt6.QtWidgets", "PyQt6.QtCore", "PyQt6.QtGui",
    ],
    excludes=[
        "onnx.reference", "onnx.reference.ops",
        "torch.distributed", "torch.testing._internal",
        "torch.utils.tensorboard", "scipy", "matplotlib",
    ],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    cipher=block_cipher, noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
    name="MyCustomOCR", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=True, console=False,
    disable_windowed_traceback=False, target_arch=None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[], name="MyCustomOCR",
)

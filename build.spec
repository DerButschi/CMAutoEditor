# -*- mode: python ; coding: utf-8 -*-

import profiles.cold_war.barns
from osm_utils import *

block_cipher = None

a_cmautoeditor = Analysis(
    ['cmautoeditor.py'],
    pathex=['osm_utils', 'profiles'],
    binaries=[],
    datas=[],
    hiddenimports=["profiles"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_dgm2cm = Analysis(
    ['dgm2cm.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=["skimage.measure", "skimage.transform", "osm_utils"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_geotiff2cm = Analysis(
    ['geotiff2cm.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=["rasterio", "rasterio.control", "rasterio.rpc", "rasterio.crs", "rasterio.sample", "rasterio.vrt", "rasterio._features", "rasterio.transform"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


MERGE((a_cmautoeditor, 'cmautoeditor', 'cmautoeditor'), (a_dgm2cm, 'dgm2cm', 'dgm2cm'), (a_geotiff2cm, 'geotiff2cm', 'geotiff2cm'))

pyz_cmautoeditor = PYZ(a_cmautoeditor.pure, a_cmautoeditor.zipped_data, cipher=block_cipher)

exe_cmautoeditor = EXE(
    pyz_cmautoeditor,
    a_cmautoeditor.dependencies,
    a_cmautoeditor.scripts,
    a_cmautoeditor.binaries,
    a_cmautoeditor.zipfiles,
    a_cmautoeditor.datas,
    [],
    name='cmautoeditor',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

pyz_dgm2cm = PYZ(a_dgm2cm.pure, a_dgm2cm.zipped_data, cipher=block_cipher)

exe_dgm2cm = EXE(
    pyz_dgm2cm,
    a_dgm2cm.dependencies,
    a_dgm2cm.scripts,
    a_dgm2cm.binaries,
    a_dgm2cm.zipfiles,
    a_dgm2cm.datas,
    [],
    name='dgm2cm',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

pyz_geotiff2cm = PYZ(a_geotiff2cm.pure, a_geotiff2cm.zipped_data, cipher=block_cipher)

exe_geotiff2cm = EXE(
    pyz_geotiff2cm,
    a_geotiff2cm.dependencies,
    a_geotiff2cm.scripts,
    a_geotiff2cm.binaries,
    a_geotiff2cm.zipfiles,
    a_geotiff2cm.datas,
    [],
    name='geotiff2cm',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)



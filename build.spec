# -*- mode: python ; coding: utf-8 -*-

import skimage

block_cipher = None


cmautoeditor_a = Analysis(['cmautoeditor.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

cmautoeditor_pyz = PYZ(cmautoeditor_a.pure, cmautoeditor_a.zipped_data,
             cipher=block_cipher)

cmautoeditor_exe = EXE(cmautoeditor_pyz,
          cmautoeditor_a.scripts, 
          [],
          exclude_binaries=True,
          name='cmautoeditor',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )

dgm2cm_a = Analysis(['data_conversion\\dgm2cm.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=[skimage.measure],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
dgm2cm_pyz = PYZ(dgm2cm_a.pure, dgm2cm_a.zipped_data,
             cipher=block_cipher)

dgm2cm_exe = EXE(dgm2cm_pyz,
          dgm2cm_a.scripts, 
          [],
          exclude_binaries=True,
          name='dgm2cm',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )

coll = COLLECT(cmautoeditor_exe,
               cmautoeditor_a.binaries,
               cmautoeditor_a.zipfiles,
               cmautoeditor_a.datas,
               dgm2cm_exe,
               dgm2cm_a.binaries,
               dgm2cm_a.zipfiles,
               dgm2cm_a.datas,

               strip=False,
               upx=True,
               upx_exclude=[],
               name='cmautoeditor_pkg')

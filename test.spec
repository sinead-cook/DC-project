# -*- mode: python -*-

block_cipher = None

import subprocess
import os

a = Analysis(['startupdlg.py'],
             pathex=['/Users/Sinead/app/'],
             binaries=[],
             datas=[],
             hiddenimports=['UserList', 'UserString'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='startupdlg',
          debug=True,
          strip=False,
          upx=True,
          console=True )


# -*- mode: python -*-

block_cipher = None


a = Analysis(['startupdlg.py'],
             pathex=['/Users/Sinead/app'],
             binaries=[],
             datas=[],
             hiddenimports=['UserString', 'UserList'],
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
          exclude_binaries=True,
          name='startupdlg',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='startupdlg')
app = BUNDLE(coll,
             name='startupdlg.app',
             icon=None,
             bundle_identifier=None)

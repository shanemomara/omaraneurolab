@echo off
C:/Users/smartin5/AppData/Local/Programs/Python/Python37/python.exe -m PyInstaller --noconfirm --log-level=WARN ^
    --onefile --nowindow ^
    --hidden-import=xlrd ^
    --hidden-import=openpyxl ^
    --name=NeuroChaT ^
    --icon=NeuroChat ^
    cli.spec
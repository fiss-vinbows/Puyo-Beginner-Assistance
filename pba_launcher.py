"""PBA.exe の入口(PyInstaller 用)。ソースから起動する時は start.bat(python -m src.app)を使う。"""

from src.app import main

if __name__ == "__main__":
    main()

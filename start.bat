@echo off
REM ぷよぷよ支援AI 起動用バッチファイル
REM ダブルクリックするだけで仮想環境を有効化してapp.pyを起動する
REM (pythonwを使い、アプリを閉じた後にコンソールが残らないようにしている)
cd /d "%~dp0"
call venv\Scripts\activate.bat
start "" venv\Scripts\pythonw.exe -m src.app

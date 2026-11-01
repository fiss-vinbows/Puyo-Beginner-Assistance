@echo off
REM ama ブリッジ(思考エンジン)のビルド。WinLibs (MinGW-w64 g++) が必要:
REM   winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --source winget
REM 生成物: external\ama-main\bin\bridge\ama-bridge.exe (静的リンク、DLL不要)
cd /d "%~dp0..\external\ama-main"
set "PATH=C:\Users\fissk\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;%PATH%"
mingw32-make PEXT=true bridge
if errorlevel 1 (echo ビルド失敗 & exit /b 1)
copy /y config.json bin\bridge\config.json >nul
echo ビルド完了: bin\bridge\ama-bridge.exe

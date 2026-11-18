@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d %~dp0
set "out=%~1"
if not defined out set "out=%CD%"
if not exist "%out%" mkdir "%out%"
nvcc -O3 -arch=sm_120 --fmad=false -prec-div=true -prec-sqrt=true -std=c++17 -Xcompiler "/fp:strict /EHsc" -Xptxas=-v -I..\..\.. bench\core_bench.cu -o "%out%\mm_fp32_core_bench.exe"
exit /b %errorlevel%

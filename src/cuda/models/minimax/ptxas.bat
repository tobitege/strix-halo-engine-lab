@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d %~dp0
nvcc -O3 -arch=sm_120 --fmad=false -prec-div=true -prec-sqrt=true -std=c++20 -Xptxas -v -I..\..\.. -c fp32_identity.cu -o %TEMP%\fp32id.obj

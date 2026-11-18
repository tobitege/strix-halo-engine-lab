@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d %~dp0
nvcc -DNOMINMAX -DWIN32_LEAN_AND_MEAN -O3 -arch=sm_120 --fmad=false -prec-div=true -prec-sqrt=true -std=c++20 -Xcompiler "/fp:strict /EHsc /O2" -I..\..\.. fp32_identity.cu mm_rtx_kernels.cu mm_rtx_prefill_main.cu mm_rtx_pack.cpp mm_rtx_weights.cpp ..\..\..\bp_codec.cpp ..\..\..\handoff.cpp -I..\..\..\..\include -lws2_32 -o mm-rtx-prefill.exe

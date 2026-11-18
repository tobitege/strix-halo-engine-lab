@echo off
rem Build one standalone CUDA Qwen prefill test: build_qwen_test.bat <worktree> <test name without .cu> <out dir> [extra .cu/.cpp sources]
rem Exactness flags: no FMA contraction, IEEE division and sqrt, no flush to zero. sm_120 (RTX PRO 6000) + sm_89 (RTX 4090).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set WT=%~1
set NAME=%~2
set OUT=%~3
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" -O3 -std=c++20 --fmad=false -prec-div=true -prec-sqrt=true -ftz=false ^
  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_89,code=sm_89 -Xcompiler "/EHsc /Zc:preprocessor /utf-8" ^
  -I%WT%\include -I%WT%\src\cuda -I%WT%\tests\cuda ^
  %WT%\tests\cuda\%NAME%.cu %WT%\src\gguf_reader.cpp %WT%\src\cuda\core\amd_math_tables.cu %4 %5 %6 %7 %8 %9 -o %OUT%\%NAME%.exe

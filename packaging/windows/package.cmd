@echo off
rem Build and package the Windows CUDA release (ZIP; plus NSIS or WiX when makensis / candle is on PATH).
rem   packaging\windows\package.cmd BUILD_DIR qwen|ds41 [CUDA_ARCHS] [TIERSTREAM_DIR]
rem Needs Visual Studio 2022 (x64 tools), CUDA 13.0 (CUDA_PATH), CMake >= 3.24 and Ninja. Nothing is downloaded.
rem ds41 also needs a tierstream checkout with its CUDA library built (docs/discrete-gpu.md).
setlocal
set "SRC=%~dp0..\.."
set "BUILD=%~1"
set "MODE=%~2"
set "ARCHS=%~3"
set "TIERSTREAM=%~4"
rem No labels or goto: cmd mis-scans labels in LF-only batch files (the repository stores text as LF).
set "USAGE=usage: package.cmd BUILD_DIR qwen-or-ds41 [CUDA_ARCHS, default 120] [TIERSTREAM_DIR, ds41 only]"
if "%BUILD%"=="" (echo %USAGE% & exit /b 2)
if "%ARCHS%"=="" set "ARCHS=120"
if "%CUDA_PATH%"=="" (echo package.cmd: CUDA_PATH is not set & exit /b 2)
where cl >nul 2>nul || call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
where cl >nul 2>nul || (echo package.cmd: no MSVC x64 compiler; run from a VS x64 developer prompt & exit /b 2)
set "COMMON=-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DBUILD_TESTING=OFF -DSTRIX_PACKAGING=ON -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON -DCMAKE_CUDA_ARCHITECTURES=%ARCHS% "-DCMAKE_CUDA_COMPILER=%CUDA_PATH:\=/%/bin/nvcc.exe""
if /i "%MODE%"=="qwen" (
  set "OPTS=-DSTRIX_CUDA_QWEN_ONLY=ON"
  set "TARGETS=strix-cuda-qwen-serve strix-cuda-qwen-prefill-service"
) else if /i "%MODE%"=="ds41" (
  if "%TIERSTREAM%"=="" (echo %USAGE% & exit /b 2)
  set "OPTS=-DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_DS41=ON -DSTRIX_FAMILY_DEEPSEEK41=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_TIERSTREAM_DIR=%TIERSTREAM%"
  set "TARGETS=strix-cuda-ds41-replica strix-cuda-ds41-decode strix-cuda-ds41-prefill strix-cuda-ds41-prefill-service strix-cuda-ds41-pipeline-front strix-cuda-ds41-multi-front"
) else (echo %USAGE% & exit /b 2)
cmake -S "%SRC%" -B "%BUILD%" %COMMON% %OPTS% > "%BUILD%-configure.log" 2>&1 || (echo configure failed: %BUILD%-configure.log & exit /b 1)
cmake --build "%BUILD%" --target %TARGETS% > "%BUILD%-build.log" 2>&1 || (echo build failed: %BUILD%-build.log & exit /b 1)
pushd "%BUILD%"
cpack > cpack.log 2>&1 || (echo cpack failed: %BUILD%\cpack.log & popd & exit /b 1)
popd
echo PACKAGE OK: %BUILD%
dir /b "%BUILD%\strix-*-windows-*.zip" "%BUILD%\strix-*-windows-*.exe" "%BUILD%\strix-*-windows-*.msi" 2>nul
exit /b 0

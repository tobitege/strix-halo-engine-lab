param([Parameter(Mandatory=$true)][string]$BuildDir,
    # CUTLASS v4.2.1 source tree (git clone --branch v4.2.1 https://github.com/NVIDIA/cutlass)
    [Parameter(Mandatory=$true)][string]$CutlassDir)
$ErrorActionPreference = 'Stop'
$cutlass = (Resolve-Path $CutlassDir).Path
if (!(Select-String -Quiet -Path (Join-Path $cutlass 'include/cutlass/version.h') -Pattern '^#define CUTLASS_MINOR 2$')) { throw 'CutlassDir is not CUTLASS 4.2.x' }
$build = [IO.Path]::GetFullPath($BuildDir)
New-Item -ItemType Directory -Force $build | Out-Null
New-Item -ItemType Directory -Force (Join-Path $build 'tmp') | Out-Null
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
$vs = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$script = @"
@echo off
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
set "TMP=$build\tmp"
set "TEMP=$build\tmp"
call "$vs" >nul
if errorlevel 1 exit /b 1
cd /d "$build"
cl /nologo /std:c++20 /O2 /EHsc /fp:strict /I"$cuda\include" "$PSScriptRoot\host_tests.cpp" /Fe:host_tests.exe /Fo:host_tests.obj
if errorlevel 1 exit /b 1
"$cuda\bin\nvcc.exe" -O3 -arch=sm_120 --fmad=false -ftz=false -prec-div=true -prec-sqrt=true -std=c++20 -I"$cutlass\include" -Xcompiler "/fp:strict /EHsc /O2 /MD /arch:AVX2" -Xptxas=-v "$PSScriptRoot\..\native.cu" "$PSScriptRoot\bench.cu" -o glm_native_bench.exe
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /O2 /EHsc /fp:strict "$PSScriptRoot\kda_host_tests.cpp" /Fe:kda_host_tests.exe /Fo:kda_host_tests.obj
if errorlevel 1 exit /b 1
"$cuda\bin\nvcc.exe" -O3 -arch=sm_120 --fmad=false -ftz=false -prec-div=true -prec-sqrt=true -std=c++20 -I"$cutlass\include" -Xcompiler "/fp:strict /EHsc /O2 /MD /arch:AVX2" -Xptxas=-v "$PSScriptRoot\..\kda.cu" "$PSScriptRoot\kda_bench.cu" -o glm_kda_bench.exe
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /O2 /EHsc /fp:strict "$PSScriptRoot\dsa_host_tests.cpp" /Fe:dsa_host_tests.exe /Fo:dsa_host_tests.obj
if errorlevel 1 exit /b 1
"$cuda\bin\nvcc.exe" -O3 -arch=sm_120 --fmad=false -ftz=false -prec-div=true -prec-sqrt=true -std=c++20 -I"$cutlass\include" -Xcompiler "/fp:strict /EHsc /O2 /MD /arch:AVX2" -Xptxas=-v "$PSScriptRoot\..\native.cu" "$PSScriptRoot\..\dsa.cu" "$PSScriptRoot\dsa_bench.cu" -o glm_dsa_bench.exe
if errorlevel 1 exit /b 1
"$cuda\bin\cuobjdump.exe" --dump-sass glm_dsa_bench.exe > glm_dsa_bench.sass
if errorlevel 1 exit /b 1
"$cuda\bin\cuobjdump.exe" --dump-ptx glm_dsa_bench.exe > glm_dsa_bench.ptx
if errorlevel 1 exit /b 1
"$cuda\bin\cuobjdump.exe" --dump-sass glm_kda_bench.exe > glm_kda_bench.sass
if errorlevel 1 exit /b 1
"$cuda\bin\cuobjdump.exe" --dump-ptx glm_kda_bench.exe > glm_kda_bench.ptx
if errorlevel 1 exit /b 1
"$cuda\bin\cuobjdump.exe" --dump-sass glm_native_bench.exe > glm_native_bench.sass
if errorlevel 1 exit /b 1
"$cuda\bin\cuobjdump.exe" --dump-ptx glm_native_bench.exe > glm_native_bench.ptx
"@
$script | Set-Content (Join-Path $build 'build.cmd') -Encoding ascii
& cmd /c (Join-Path $build 'build.cmd') 2>&1 | Tee-Object (Join-Path $build 'build.log')
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$sass = Get-Content (Join-Path $build 'glm_native_bench.sass') -Raw
if ($sass -notmatch 'HMMA\.') { throw 'No HMMA instruction in emitted SASS' }
& python -B (Join-Path $PSScriptRoot 'inspect_sass.py') (Join-Path $build 'glm_native_bench.sass') --out (Join-Path $build 'sass-evidence.json')
if ($LASTEXITCODE -ne 0) { throw 'Inner-loop/reference SASS check failed' }
& python -B (Join-Path $PSScriptRoot 'inspect_kda_sass.py') (Join-Path $build 'glm_kda_bench.sass') --out (Join-Path $build 'kda-sass-evidence.json')
if ($LASTEXITCODE -ne 0) { throw 'KDA projection/chunk/reference SASS check failed' }
& python -B (Join-Path $PSScriptRoot 'inspect_dsa_sass.py') (Join-Path $build 'glm_dsa_bench.sass') --out (Join-Path $build 'dsa-sass-evidence.json')
if ($LASTEXITCODE -ne 0) { throw 'DSA/MTP SASS check failed' }
Get-FileHash (Join-Path $build 'glm_dsa_bench.exe'),(Join-Path $build 'dsa_host_tests.exe'),(Join-Path $build 'glm_native_bench.exe'),(Join-Path $build 'host_tests.exe'),(Join-Path $build 'glm_kda_bench.exe'),(Join-Path $build 'kda_host_tests.exe') -Algorithm SHA256
# Compilation and disassembly only. Run host_tests.exe separately; never launch
# the GPU benchmark from a build script.

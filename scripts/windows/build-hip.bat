@echo off
rem Native Windows HIP build of the Qwen single-node engine (docs/getting-started/windows.md).
rem Compiler: clang++ of the ROCm SDK for Windows inside the Visual Studio build environment, with Ninja.
rem
rem usage: build-hip.bat [configure|build|all] [target ...]
rem   targets default to strix-hip-decode-token
rem environment:
rem   ROCM_SDK           the SDK folder that holds bin\ and lib\llvm\bin\clang++.exe (required)
rem   VCVARS             vcvars64.bat of a Visual Studio with the MSVC 14.4x toolset; found through vswhere
rem                      when unset (Visual Studio 2022, else a newer one with the 14.44 component)
rem   VCVARS_ARGS        arguments for it, for example -vcvars_ver=14.44; set by that search when needed
rem   STRIX_BUILD_DIR    build folder, default <repository>\build-hip
rem   STRIX_HIP_ARCH     HIP target, default gfx1151
rem   STRIX_EXTRA_CMAKE  more configure options, for example -DBUILD_TESTING=ON
rem                      (cmd splits arguments at "=", so they come from the environment)
rem
rem Two settings differ from the Linux build:
rem   STRIX_WARNINGS_AS_ERRORS=OFF  the Linux build compiles the C++ sources with GCC; clang reports more
rem                                 implicit conversions in them (none of them a 32-bit long, see the page above)
rem   _CRT_SECURE_NO_WARNINGS       the Windows C runtime marks getenv, fopen and similar as deprecated
setlocal
set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
set "BUILD=%REPO%\build-hip"
if not "%STRIX_BUILD_DIR%"=="" set "BUILD=%STRIX_BUILD_DIR%"
set "ARCH=gfx1151"
if not "%STRIX_HIP_ARCH%"=="" set "ARCH=%STRIX_HIP_ARCH%"

if "%ROCM_SDK%"=="" (echo ROCM_SDK is not set & exit /b 2)
if not exist "%ROCM_SDK%\lib\llvm\bin\clang++.exe" (echo no clang++ under %ROCM_SDK%\lib\llvm\bin & exit /b 2)
set "SDK=%ROCM_SDK%"
set "SDKF=%ROCM_SDK:\=/%"

rem The C++ toolset must be MSVC 14.4x, the one of Visual Studio 2022. The C++ library of 14.5x (Visual Studio
rem 2026) does not compile with the HIP headers of ROCm SDK 7.14: its cmath comparison functions clash with the
rem SDK's device declarations. Visual Studio 2022 has 14.4x as its own toolset. A newer Visual Studio can carry
rem it as the component MSVC v143 14.44, which vcvars selects with -vcvars_ver.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if "%VCVARS%"=="" for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -version [17.0^,18.0^) -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
if "%VCVARS%"=="" for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64 -property installationPath`) do (
    set "VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
    set "VCVARS_ARGS=-vcvars_ver=14.44"
)
if not exist "%VCVARS%" (echo no Visual Studio with the MSVC 14.4x toolset found; set VCVARS to its vcvars64.bat & exit /b 2)
call "%VCVARS%" %VCVARS_ARGS% >nul || exit /b 2
if not "%VCToolsVersion:~0,4%"=="14.4" (echo the C++ toolset is %VCToolsVersion%, MSVC 14.4x is required & exit /b 2)

set "HIP_PLATFORM=amd"
set "HIP_COMPILER=clang"
set "HIP_RUNTIME=rocclr"
set "ROCM_PATH=%SDK%"
set "HIP_PATH=%SDK%"
set "HIP_DEVICE_LIB_PATH=%SDK%\lib\llvm\amdgcn\bitcode"
set "PATH=%SDK%\bin;%SDK%\lib\llvm\bin;%PATH%"

set "TARGETS="
:collect
shift
if "%~1"=="" goto collected
set "TARGETS=%TARGETS% %~1"
goto collect
:collected
if "%TARGETS%"=="" set "TARGETS=strix-hip-decode-token"

if /i "%MODE%"=="build" goto build
if /i not "%MODE%"=="configure" if /i not "%MODE%"=="all" (echo usage: build-hip.bat [configure^|build^|all] [target ...] & exit /b 2)
cmake -S "%REPO%" -B "%BUILD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER=%SDKF%/lib/llvm/bin/clang++.exe ^
  -DCMAKE_HIP_COMPILER=%SDKF%/lib/llvm/bin/clang++.exe ^
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=%SDKF% ^
  -DCMAKE_HIP_ARCHITECTURES=%ARCH% ^
  "-DCMAKE_CXX_FLAGS=-D_CRT_SECURE_NO_WARNINGS" ^
  "-DCMAKE_HIP_FLAGS=-D_CRT_SECURE_NO_WARNINGS --rocm-path=%SDKF% --rocm-device-lib-path=%SDKF%/lib/llvm/amdgcn/bitcode" ^
  -DSTRIX_WARNINGS_AS_ERRORS=OFF ^
  -DSTRIX_ENABLE_HIP_RUNTIME=ON ^
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON ^
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON ^
  -DSTRIX_USE_OPENSSL_SHA256=OFF ^
  -DBUILD_TESTING=OFF %STRIX_EXTRA_CMAKE%
if errorlevel 1 (echo configure failed & exit /b 3)
if /i "%MODE%"=="configure" exit /b 0

:build
rem -k 0: keep going after a failed file, so one run lists every error
ninja -C "%BUILD%" -k 0 %TARGETS%
if errorlevel 1 (echo build failed & exit /b 4)
rem Windows loads a program's DLLs from its own folder before System32, where the display driver's HIP
rem runtime sits (a different version: it fails at the first kernel launch). The SDK's runtime goes next
rem to the programs.
for %%P in (amdhip64*.dll amd_comgr*.dll rocm_kpack*.dll) do copy /y "%SDK%\bin\%%P" "%BUILD%\" >nul
exit /b 0

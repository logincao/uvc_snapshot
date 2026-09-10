@echo off
setlocal
set "VCVARS=X:\vs2026\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=T:\Program Files\CMake\bin\cmake.exe"
set "NINJA=%CD%\tools\ninja.exe"
if not exist "%VCVARS%" goto novc
if not exist "%NINJA%" goto noninja
call "%VCVARS%"
if not "%ERRORLEVEL%"=="0" goto loadfail
"%CMAKE%" -S "%CD%" -B "%CD%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_MAKE_PROGRAM=%NINJA%" >nul
if not "%ERRORLEVEL%"=="0" goto configurefail
"%CMAKE%" --build "%CD%\build" --parallel
rem NOTE: cmake/link may return a negative code (e.g. -1); "if errorlevel 1" tests >= 1
rem and would MISS it, so always compare against 0 instead.
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto compilefail
echo BUILD_OK main.exe
goto end
:novc
echo ERROR_vcvars_not_found
exit /b 1
:noninja
echo ERROR_ninja_not_found
exit /b 1
:loadfail
echo ERROR_load_env_failed
exit /b 1
:configurefail
echo ERROR_configure_failed
exit /b 1
:compilefail
echo ERROR_compile_failed rc=%RC%
exit /b 1
:end
endlocal

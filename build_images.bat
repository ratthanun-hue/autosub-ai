@echo off
rem ==============================================================================
rem AutoSub-AI Dual-Image Build Tool (Windows Batch)
rem Builds Decoupled Docker Images:
rem   1. Dynamic Node (Audio Worker: Steps 0-5) -> Dockerfile.dynamic
rem   2. Master Node (LLM Master: Step 6)       -> Dockerfile.master
rem ==============================================================================

set REGISTRY=%1
if "%REGISTRY%"=="" set REGISTRY=anakim

set TAG=%2
if "%TAG%"=="" set TAG=latest

echo ==========================================================
echo === 1/2 BUILDING DYNAMIC WORKER (STEPS 0-5 AUDIO) ===
echo ==========================================================
docker build -f Dockerfile.dynamic -t "%REGISTRY%/autosub-worker:%TAG%" .
if errorlevel 1 goto error

echo ==========================================================
echo === 2/2 BUILDING MASTER LLM (STEP 6 QWEN2.5) ===
echo ==========================================================
docker build -f Dockerfile.master -t "%REGISTRY%/autosub-master:%TAG%" .
if errorlevel 1 goto error

echo ==========================================================
echo === BUILD COMPLETE ===
echo Images created:
echo   - %REGISTRY%/autosub-worker:%TAG%
echo   - %REGISTRY%/autosub-master:%TAG%
echo ==========================================================
echo To push to Docker Hub, run:
echo   docker push %REGISTRY%/autosub-worker:%TAG%
echo   docker push %REGISTRY%/autosub-master:%TAG%
echo ==========================================================
goto end

:error
echo [ERROR] Build failed! Check the Docker output above.
exit /b 1

:end

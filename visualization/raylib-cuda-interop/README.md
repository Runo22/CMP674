# Raylib + CUDA-OpenGL Interop Visualizer

This branch adds a live CUDA visualization built with [raylib](https://github.com/raysan5/raylib). Raylib creates the window, handles input, and draws text overlays. CUDA computes the simulation and collision flags, maps the OpenGL vertex buffer, and writes particle render data directly into it.

There is no per-frame CUDA-to-CPU particle copy. The visual path is:

```text
raylib window/context
        |
        v
OpenGL VBO created with rlgl
        |
        v
CUDA maps the VBO with cudaGraphicsGLRegisterBuffer
        |
        v
CUDA writes positions/colors/sizes
        |
        v
raylib/rlgl draws the same VBO
```

## Requirements

- Windows with NVIDIA GPU
- NVIDIA driver
- CUDA Toolkit
- CMake
- Visual Studio 2022 or Build Tools
- vcpkg
- raylib

Install raylib with vcpkg:

```powershell
vcpkg install raylib:x64-windows
```

## Build

```powershell
cd visualization/raylib-cuda-interop
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
.\Release\raylib_cuda_visualizer.exe 2500
```

## Controls

- `Space`: pause/resume
- `C`: switch uniform/clustered distribution
- `R`: reset current distribution
- `Esc`: quit

## Current Scope

The first Raylib interop version visualizes a live CUDA brute force collision kernel:

- blue particles are not colliding
- red particles are colliding
- metrics are displayed in the top-left overlay

The next planned extension is a CUDA uniform-grid visualization mode with grid overlays and candidate-pair sampling.

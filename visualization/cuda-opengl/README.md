# CUDA OpenGL Live Collision Visualizer

This is a Windows/NVIDIA-focused live visualization target. CUDA owns the simulation state, computes collisions every frame, maps an OpenGL VBO with CUDA-OpenGL interop, and writes render vertices directly into that VBO. OpenGL only draws the buffer.

## Requirements

- NVIDIA GPU and driver
- CUDA Toolkit
- CMake
- Visual Studio 2022 or Build Tools
- GLFW
- GLEW

The easiest dependency path on Windows is vcpkg:

```powershell
vcpkg install glfw3:x64-windows glew:x64-windows
```

## Build

From this directory:

```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
.\Release\cuda_collision_visualizer.exe 2500
```

## Controls

- `Space`: pause/resume
- `C`: switch uniform/clustered distribution
- `R`: reset current distribution
- `Esc`: quit

## What Is Actually Visualized?

The displayed particles come from a real CUDA kernel. Each frame:

1. CUDA updates particle positions.
2. CUDA brute force collision detection marks colliding particles.
3. CUDA maps the OpenGL VBO.
4. CUDA writes particle positions and colors directly into the VBO.
5. OpenGL draws the VBO.

Blue particles are not colliding. Red particles are colliding.

This first interop version visualizes the CUDA brute force kernel. The next extension can add a CUDA uniform-grid visualizer path with grid-cell VBO overlays and candidate-pair sampling.

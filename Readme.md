# Vulkan 1.3 Minimal Example
---

Aim of this project is to create a purely minimal example of using Vulkan-Hpp, Vulkan Memory Allocator,
vk-bootstrap and glm.

It does not use any "framework" style scaffolding.

## Project Setup
Project uses CMake `3.31.x`, Ninja `1.12.1` and VCPKG, for a cross-platform build (Windows and Linux)
`C++23` support is required, including C++ Modules.
Vulkan SDK and vk-bootstrap are pegged to Vulkan `1.3.296`.

For Linux build, it requires LLVM/Clang version `19.1.x` or better.
For Windows build, it requires MSVC version `17.10.x` or better.

There are two presets in the config, one for windows and one for linux

### Dependencies
- Vulkan SDK `1.3.296`
- VK-Bootstrap `1.3.296`
- Vulkan Memory Allocator `3.1`
- Vulkan Memory Allocator HPP `3.1`
- GLM
- GLFW `3.x`

## Code Layout
All the code is in `vk-min-src.cpp` file. 
Have made best effort to document/comment how it's being used.
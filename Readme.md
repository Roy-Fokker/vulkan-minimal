# Vulkan 1.3 Minimal Example
---

Aim of this project is to create a purely minimal example of using Vulkan-Hpp, Vulkan Memory Allocator,
vk-bootstrap and glm.

It does not use any "framework" style scaffolding.

## Project Setup
Project uses CMake `3.31.x`, Ninja `1.12.1` and VCPKG, for a cross-platform build (Windows and Linux)
`C++23` support is required, including C++ Modules.
Vulkan SDK is at `1.3.296`, which was latest available SDK, and vk-bootstrap is at `1.3.298`.

For Linux build, it requires LLVM/Clang version `18.1.x` or better.
For Windows build, it requires MSVC version `17.10.x` or better.

There are two presets in the config, one for windows and one for linux.
Both builds assume Vulkan sdk is setup correctly.

### Dependencies
- Vulkan SDK `1.3.296`
- DXC, HLSL shader compiler from Vulkan SDK
- VK-Bootstrap `1.3.298`, retrived via vcpkg overlay from vk-bootstrap github.
- Vulkan Memory Allocator `3.1`, used by VMA-HPP
- Vulkan Memory Allocator HPP `3.1`
- GLM
- GLFW `3.x`
- dds-ktx, retrived via vcpkg overlay from dds-ktx github

## Code Layout
All the code is in `vk-min-src.cpp` file. 
Have made best effort to document/comment how it's being used.

## Vulkan features used
- Viewport height negative, to get +X right, +Y up, +Z into screen (LHS)
- Dynamic Rendering
- Shader Device Address
- Descriptor Buffers
  - Uniform buffer
  - Texture buffer, image + sampler

## References
- https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/descriptor_buffer_basic
- https://github.com/steps3d/vulkan-with-classes/blob/main/example-descriptor-buffer.cpp
- https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_descriptor_buffer.html
- https://www.khronos.org/blog/vk-ext-descriptor-buffer

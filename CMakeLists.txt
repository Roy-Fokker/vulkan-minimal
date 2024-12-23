cmake_minimum_required(VERSION 3.31)

# Enable scan for dependencies
set(CXX_SCAN_FOR_MODULES ON)
# disable C++ extensions
set(CMAKE_CXX_EXTENSIONS OFF)
# force C++ version to 23
set(CMAKE_CXX_STANDARD 23)
# standard is required
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the experimental flag
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
	"0e5b6991-d74f-4b3d-a41c-cf096e0b2508" # current value as per https://www.kitware.com/import-std-in-cmake-3-30/
)

# tell CMake we want to use 'import std' 
# will get enabled for all targets declared after this
set(CMAKE_CXX_MODULE_STD 1)

# project and language to be used
project(vulkan-minimal
	LANGUAGES CXX
	DESCRIPTION "C++23 based Minimal Vulkan Example.")

#---------------------------------------------------------------------------------------

set (VULKAN_COMPONENTS "dxc" "dxc_exe" "glslc")                      # Componenets we need CMake to ensure exist
find_package(Vulkan REQUIRED COMPONENTS ${VULKAN_COMPONENTS})        # Vulkan SDK
find_package(VulkanMemoryAllocator CONFIG REQUIRED)                  # Vulkan Memory Allocator
find_package(unofficial-vulkan-memory-allocator-hpp CONFIG REQUIRED) # Vulkan Memory Allocator - Hpp
find_package(vk-bootstrap CONFIG REQUIRED)                           # vk-bootstrap
find_package(glm CONFIG REQUIRED)                                    # GLM math library
find_package(glfw3 CONFIG REQUIRED)                                  # GLFW library

#---------------------------------------------------------------------------------------

# function to compile HLSL into SPIR-V files
function(target_hlsl_sources TARGET)
	if (NOT TARGET Vulkan::dxc_exe)
		message(FATAL_ERROR "[Error]: Could not find dxc")
	endif()

	# figure out how many files we have to configure given the pattern
	list(LENGTH ARGN count_HLSL)
	math(EXPR count_HLSL "${count_HLSL} / 3")

	# List of compiled shader output
	set(shader_files "")
	set(shader_sources "")

	# Loop through all the pairs for filename:profile provided
	foreach(i RANGE 1 ${count_HLSL})
		math(EXPR fni "(${i}-1)*3")              # filename index
		math(EXPR pfi "${fni}+2")                # profile index
		list(GET ARGN ${fni} hlsl_filename)      # get the filename[i]
		list(GET ARGN ${pfi} hlsl_profile)       # get the profile[i]

		# get the absolute path of current source file
		file(REAL_PATH ${hlsl_filename} source_abs)

		if(NOT EXISTS ${source_abs})
			message(FATAL_ERROR "Cannot find shader file: ${source_abs}")
		endif()

		# get only the filename from absolute path
		cmake_path(GET source_abs STEM basename)
		set(basename "${basename}.${hlsl_profile}")

		# get only the parent directory of the file from absolute path
		cmake_path(GET source_abs PARENT_PATH source_fldr)
		get_filename_component(source_fldr "${source_fldr}" NAME)
		
		# shader output folder will be a subfolder in the binary directory
		set(shader_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)

		# full path to compiled output 
		set(output ${shader_dir}/${basename}.spv)

		# call vulkan sdk's dxc compiler with source and output arguments.
		add_custom_command(
			OUTPUT ${output}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${shader_dir}
			COMMAND Vulkan::dxc_exe -spirv -E main -Fo ${output} -T ${hlsl_profile} ${source_abs}
			DEPENDS ${source_abs}
			COMMENT "DXC Compiling SPIRV: ${hlsl_filename} -> ${output}"
			VERBATIM
		)

		list(APPEND shader_sources "${hlsl_filename}")
		list(APPEND shader_files "${output}")
	endforeach()

	# make a new variable to hold all output target names
	set(shader_group "${TARGET}_HLSL")
	# add custom target using new variable bound to output file of glslc step
	add_custom_target("${shader_group}"
					  DEPENDS "${shader_files}"
					  SOURCES "${shader_sources}"
	)
	# add compilation of this shader as dependency of the target
	add_dependencies("${TARGET}" "${shader_group}")
endfunction()

#---------------------------------------------------------------------------------------

# set platform specific configurations
if(WIN32)
	# Set Properties for Windows specific configurations
	list(APPEND platform_definitions
		_CRT_SECURE_NO_WARNINGS     # Don't want MSVC's CRT warnings
		NOMINMAX                    # Don't want Window.h to define min and max macros
		WIN32_LEAN_AND_MEAN         # Windows.h should only include the basics
	)

	# Set Compiler flags for MSVC
	# Not used atm, as preset set same options for msvc
	list(APPEND compiler_options
		/utf-8                 # Set source and execution character sets to UTF-8
		/EHsc                  # Enable standard C++ stack unwinding, assume C code doesn't throw exceptions
		/W4                    # Enable all the warnings
		/permissive-           # Set standards conformance mode
		/Zc:__cplusplus        # Enable the __cplusplus macro to report the supported standard
		/Zc:preprocessor       # Use the new conforming preprocessor
		/Zc:throwingNew        # Assume operator new throws on failure
		/Zc:inline             # Remove unreferenced functions or data if they're COMDAT or have internal linkage only
		/Zc:externConstexpr    # Enable external linkage for constexpr variables
		/Zc:templateScope      # Enforce Standard C++ template parameter shadowing rules
		/Zc:checkGwOdr         # Enforce Standard C++ ODR violations under /Gw
		/Zc:enumTypes          # Enable Standard C++ rules for enum type deduction
	)

	# Set Executable Compiler Options
	list(APPEND executable_link_options
		/SUBSYSTEM:WINDOWS     # Windows UI application, not console
		/ENTRY:mainCRTStartup  # Use main as program entry point
	)
elseif(LINUX) # Linux specific configurations only works with clang/llvm
	# Set Compiler flags for clang
	# Not used atm, as preset set same options for msvc
	list(APPEND compiler_options
		-stdlib=libc++                     # use LibC++ as GCC doesn't support modules
	)
endif()

# set properties for Vulkan HPP 
list(APPEND vulkan_definitions
	VULKAN_HPP_NO_CONSTRUCTORS            # Vulkan.hpp should not define constructors for structs
	VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1  # Enable dynamic dispatch loader for EXT funcs
	VULKAN_HPP_STORAGE_SHARED             # The storage for the DispatchLoaderDynamic should be embedded in a DLL
	VULKAN_HPP_STORAGE_SHARED_EXPORT      # Export the required symbols.
)

# Set properties for GLM 
list(APPEND glm_definitions
	GLM_FORCE_DEPTH_ZERO_TO_ONE # GLM clip space should be in Z-axis to 0 to 1
	GLM_FORCE_LEFT_HANDED       # GLM should use left-handed coordinates, +z goes into screen
	GLM_FORCE_RADIANS           # GLM should always use radians not degrees.
)

# Set properties for GLFW
list(APPEND glfw_definitions
)

#---------------------------------------------------------------------------------------

# name of this application/library
set(PRJ_APP_NAME "vk-min-app")

# create executable for initial setup
add_executable(${PRJ_APP_NAME})

# ensure it's Standard C++ 23
target_compile_features(${PRJ_APP_NAME} PRIVATE cxx_std_23)

# set preprocessor defines
target_compile_definitions(${PRJ_APP_NAME}
	PRIVATE 
		UNICODE _UNICODE            # Tell compiler we are using UNICODE
		${platform_definitions}     # Get platform specific definitions
		${vulkan_definitions}       # Plaform specific Vulkan defines
		${glm_definitions}          # GLM library configuration
		${glfw_definitions}         # GLFW library configuration
)

# source files for this application
target_sources(${PRJ_APP_NAME}
	# Non C++ module source files
	PRIVATE
		vk-min-src.cpp
)

# libraries used by this application
target_link_libraries(${PRJ_APP_NAME}
	PRIVATE
		Vulkan::Vulkan                                                   # Vulkan SDK
		GPUOpen::VulkanMemoryAllocator                                   # Vulkan Memory Allocator
		unofficial::VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp # Vulkan Memory Allocator - Hpp
		vk-bootstrap::vk-bootstrap                                       # vk-bootstrap
		vk-bootstrap::vk-bootstrap-compiler-warnings                     # vk-bootstrap with compiler warnings
		glm::glm                                                         # GLM math library
		glfw                                                             # GLFW library
)

# shader source files used by this application
target_hlsl_sources(${PRJ_APP_NAME}
		basic_shader.vs.hlsl : vs_6_4
		basic_shader.ps.hlsl : ps_6_4
)

#---------------------------------------------------------------------------------------


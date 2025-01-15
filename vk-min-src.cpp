// Include files must be before the import statement
// Otherwise the compiler will complain.

// For assert macro
#include <cassert>

// Vulkan HPP header
#include <vulkan/vulkan.hpp>

// Vulkan Bootstrap header
#include <VkBootstrap.h>

// GLFW header
#include <GLFW/glfw3.h>

// GLM headers
#include <glm/glm.hpp> // Required for glm::mat4
#include <glm/ext.hpp> // Required for glm::perspective function

// Vulkan Memory Allocator HPP header/implementation
// VMA_IMPLEMENTATION must be defined in exactly one translation unit
// doesn't matter for this example, but it's good practice to define it in the same file as the implementation
#define VMA_IMPLEMENTATION
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>

// DDS and KTX texture file loader
// Similar to VMA, DDSKTX_IMPLEMENT must be defined in exactly one translation unit
// doesn't matter for this example, but it's good practice to define it in the same file as the implementation
#define DDSKTX_IMPLEMENT
#include "dds-ktx.h"

// C++23 Standard Module import
import std;

// to use 'sv', 'u' and other literals
using namespace std::literals;

/*
 * Colors for the console
 * List of colors that can be used in the console using ANSI escape codes
 * Only works on terminals that support ANSI escape codes
 * Foreground colors only, 8 normal, 8 bright/bold
 */
namespace CLR
{
	// Regular Colors
	constexpr auto BLK = "\033[0;30m";
	constexpr auto RED = "\033[0;31m";
	constexpr auto GRN = "\033[0;32m";
	constexpr auto YEL = "\033[0;33m";
	constexpr auto BLU = "\033[0;34m";
	constexpr auto MAG = "\033[0;35m";
	constexpr auto CYN = "\033[0;36m";
	constexpr auto WHT = "\033[0;37m";

	// Bright/Bold Colors
	constexpr auto BBLK = "\033[1;30m";
	constexpr auto BRED = "\033[1;31m";
	constexpr auto BGRN = "\033[1;32m";
	constexpr auto BYEL = "\033[1;33m";
	constexpr auto BBLU = "\033[1;34m";
	constexpr auto BMAG = "\033[1;35m";
	constexpr auto BCYN = "\033[1;36m";
	constexpr auto BWHT = "\033[1;37m";

	// Reset Color and Style
	constexpr auto RESET = "\033[0m";
}

/*
 * Platform IO helpers
 */
namespace io
{
	// Simple function to read a file in binary mode.
	auto read_file(const std::filesystem::path &filename) -> std::vector<std::byte>
	{
		std::println("{}Reading file: {}{}", CLR::BBLU, filename.string(), CLR::RESET);

		auto file = std::ifstream(filename, std::ios::in | std::ios::binary);

		assert(file.good() and "failed to open file!");

		auto file_size = std::filesystem::file_size(filename);
		auto buffer    = std::vector<std::byte>(file_size);

		file.read(reinterpret_cast<char *>(buffer.data()), file_size);

		file.close();

		return buffer;
	}

	// Convience alias for a span of bytes
	using byte_span  = std::span<const std::byte>;
	using byte_spans = std::span<byte_span>;

	// Convert any object type to a span of bytes
	auto as_byte_span(const auto &src) -> byte_span
	{
		return std::span{
			reinterpret_cast<const std::byte *>(&src),
			sizeof(src)
		};
	}

	// Covert a any contiguous range type to a span of bytes
	auto as_byte_span(const std::ranges::contiguous_range auto &src) -> byte_span
	{
		auto src_span   = std::span{ src };      // convert to a span,
		auto byte_size  = src_span.size_bytes(); // so we can get size_bytes
		auto byte_start = reinterpret_cast<const std::byte *>(src.data());
		return { byte_start, byte_size };
	}

	// void pointer offset
	auto offset_ptr(void *ptr, std::ptrdiff_t offset) -> void *
	{
		return reinterpret_cast<std::byte *>(ptr) + offset;
	}

	// structure to hold file data in memory
	struct texture
	{
		struct sub_data
		{
			uint32_t layer_idx;
			uint32_t mip_idx;
			uint32_t offset;
			uint32_t width;
			uint32_t height;
		};

		ddsktx_texture_info header_info;
		std::vector<sub_data> sub_info;
		std::vector<std::byte> data;
	};

	// Load DDS texture and create a vulkan image
	auto load_texture(const std::filesystem::path &filename) -> texture
	{
		auto texture_file_data = read_file(filename);

		auto texture_info      = ddsktx_texture_info{};
		auto texture_parse_err = ddsktx_error{};

		// Parse the DDS file
		auto result = ddsktx_parse(
			&texture_info,
			texture_file_data.data(),
			static_cast<int>(texture_file_data.size()),
			&texture_parse_err);
		assert(result == true and "Failed to parse texture file data");

		auto layer_rng = std::views::iota(0, texture_info.num_layers);
		auto mip_rng   = std::views::iota(0, texture_info.num_mips);
		auto img_str   = static_cast<void *>(texture_file_data.data());

		auto sub_data_rng = std::views::cartesian_product(layer_rng, mip_rng) // cartesian product will produce a pair-wise range
		                  | std::views::transform([&](auto &&layer_mip_nums) -> texture::sub_data {
			auto [layer_idx, mip_idx] = layer_mip_nums;

			auto sub_data = ddsktx_sub_data{};
			ddsktx_get_sub(
				&texture_info,
				&sub_data,
				texture_file_data.data(),
				static_cast<int>(texture_file_data.size()),
				layer_idx,
				0,
				mip_idx);

			// Get distance from start of image data minus the header offset
			auto sub_offset = static_cast<uint32_t>(uintptr_t(sub_data.buff) - uintptr_t(img_str) - texture_info.data_offset);

			return texture::sub_data{
				.layer_idx = static_cast<uint32_t>(layer_idx),
				.mip_idx   = static_cast<uint32_t>(mip_idx),
				.offset    = sub_offset,
				.width     = static_cast<uint32_t>(sub_data.width),
				.height    = static_cast<uint32_t>(sub_data.height),
			};
		});

		// Get Sub Data for this texture, convert range into vector
		auto texture_sub_data = sub_data_rng | std::ranges::to<std::vector>();

		// Remove header data from file_data in-memory.
		auto img_start_itr = std::begin(texture_file_data);
		texture_file_data.erase(img_start_itr, img_start_itr + texture_info.data_offset);
		texture_file_data.shrink_to_fit();

		std::println("\tImage has {} layer(s), and {} mipmap level(s)", texture_info.num_layers, texture_info.num_mips);
		return { texture_info, texture_sub_data, texture_file_data };
	}
}

/*
 * GLFW helpers
 * Functions to create a window using and close it on escape
 * And report GLFW errors to the console
 */
namespace glfw
{
	// Deleter for GLFWWindow to use with UniquePtr
	// As GLFWwindow is a C-style struct, it needs a custom deleter to be used with std::unique_ptr
	// This deleter will destroy the window and terminate GLFW
	struct destroy_glfw_win
	{
		void operator()(GLFWwindow *ptr)
		{
			std::println("{}Destroying window...{}", CLR::BLU, CLR::RESET);
			glfwDestroyWindow(ptr);
			glfwTerminate();
		}
	};

	// Callback for GLFW errors
	void error_callback(int error, const char *description)
	{
		std::println("{}Error {}:{} {}", CLR::BRED, error, CLR::RESET, description);
	}

	// Close the window when the escape key is pressed
	void close_window_on_escape(GLFWwindow *window)
	{
		glfwSetKeyCallback(window,
		                   [](GLFWwindow *window,
		                      int key,
		                      [[maybe_unused]] int scancode,
		                      int action,
		                      [[maybe_unused]] int mods) {
			if (key == GLFW_KEY_ESCAPE and action == GLFW_PRESS)
			{
				std::println("{}Escape key pressed. Closing window...{}", CLR::YEL, CLR::RESET);
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}
		});
	}

	// Create a window with GLFW
	// Initialize GLFW, create a window, set the error and escape-key-press callbacks
	auto make_window(int width, int height, std::string_view title) -> std::unique_ptr<GLFWwindow, destroy_glfw_win>
	{
		std::println("{}Creating window...{}", CLR::BLU, CLR::RESET);

		glfwSetErrorCallback(error_callback); // in case of issues print to console

		glfwInit(); // Initialize GLFW

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No OpenGL, we're using Vulkan
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // Window is not resizable, because
		                                              // example does not handle resize of Vulkan Surface/Swapchain

		auto window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr); // Create the window
		assert(window != nullptr and "Failed to create GLFW window");

		glfw::close_window_on_escape(window); // Close the window on escape

		return std::unique_ptr<GLFWwindow, destroy_glfw_win>(window); // Return the window
	}
}

/*
 * Required for Dynamic Dispatch Loader in Vulkan-HPP
 * Must be called EXACTLY once for Dynamic Dispatch Loader to work
 * Have to use Dynamic Dispatch Loader to use Descriptor Buffer Extension
 */
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

/*
 * Base Vulkan Objects
 * Everything needed to get Vulkan going.
 */
namespace base
{
	// Do we want to use Vulkan validation layers?
	// Only in Debug mode
	constexpr auto use_vulkan_validation_layers{
#ifdef _DEBUG
		true
#else
		false
#endif
	};

	// Callback for Vulkan validation layer messages
	auto debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	                    VkDebugUtilsMessageTypeFlagsEXT messageType,
	                    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
	                    [[maybe_unused]] void *pUserData)
		-> vk::Bool32
	{
		auto severity = vkb::to_string_message_severity(messageSeverity);
		auto type     = vkb::to_string_message_type(messageType);

		auto msgclr = ""sv;
		switch (messageSeverity)
		{
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
			msgclr = CLR::BRED;
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
			msgclr = CLR::BYEL;
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
			msgclr = CLR::GRN;
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
			msgclr = CLR::WHT;
			break;
		default:
			msgclr = CLR::RED;
			break;
		}

		std::println("{}[{}: {}]: {}{}", msgclr, severity, type, pCallbackData->pMessage, CLR::RESET);

		return VK_FALSE;
	}

	// Structure to hold GPU queue information
	struct queue
	{
		uint32_t family;
		vk::Queue queue;
	};

	// Structure to hold synchronization objects for each swapchain images
	struct synchronization
	{
		vk::Semaphore image_available;
		vk::Semaphore render_finished;
		vk::Fence in_flight_fence;
	};

	// All the Vulkan structures that will be used in this example
	struct vulkan_context
	{
		// Created by create_instance
		vk::Instance instance;
		vk::DebugUtilsMessengerEXT debug_messenger;

		// Created by create_surface
		vk::SurfaceKHR surface;

		// Created by pick_gpu_and_queues
		vk::PhysicalDevice chosen_gpu;
		vk::Device device;
		queue gfx_queue;
		queue transfer_queue;
		queue present_queue;
		queue compute_queue;

		// Created by create_gpu_mem_allocator
		vma::Allocator mem_allocator;

		// Created by create_swapchain
		vk::SwapchainKHR swap_chain;
		vk::Extent2D sc_extent;
		vk::Format sc_format;
		std::vector<vk::Image> sc_images;
		std::vector<vk::ImageView> sc_views;
		uint32_t max_frame_count;
		uint32_t current_frame = 0;

		// Created by create_sync_objects
		std::vector<synchronization> image_signals;
		vk::Fence tfr_in_flight_fence;

		// Created by create_command_pool
		vk::CommandPool gfx_command_pool;
		std::vector<vk::CommandBuffer> gfx_command_buffers;

		// Created by create_command_pool
		vk::CommandPool tfr_command_pool;
		vk::CommandBuffer tfr_command_buffer;
	};

	// Create a Vulkan Instance using VK-Bootstrap
	auto create_instance(vulkan_context &ctx) -> vkb::Instance
	{
		auto builder = vkb::InstanceBuilder{};

		auto vkb_ib_ret = builder.set_app_name("vk-minimal-example")
		                      .request_validation_layers(use_vulkan_validation_layers)
		                      .set_debug_callback(debug_callback)
		                      .require_api_version(1, 3, 0)
		                      .build();
		assert(vkb_ib_ret.has_value() == true and "Failed to create Vulkan Instance");

		auto vkb_inst = vkb_ib_ret.value();

		ctx.instance        = vkb_inst.instance;
		ctx.debug_messenger = vkb_inst.debug_messenger;

		VULKAN_HPP_DEFAULT_DISPATCHER.init();             // Init base for Dynamic Dispatch
		VULKAN_HPP_DEFAULT_DISPATCHER.init(ctx.instance); // Get all the other function pointers

		std::println("{}Instance created. \n"
		             "Debug Messenger set.{}",
		             CLR::GRN, CLR::RESET);

		return vkb_inst;
	}

	// Create a Vulkan Surface using GLFW
	// GLFW abstracts away the platform specific code for Vulkan Surface creation
	void create_surface(vulkan_context &ctx, GLFWwindow *window)
	{
		auto result = static_cast<vk::Result>(
			glfwCreateWindowSurface(ctx.instance,
		                            window,
		                            nullptr,
		                            reinterpret_cast<VkSurfaceKHR *>(&ctx.surface)));
		assert(result == vk::Result::eSuccess and "Failed to create Vulkan Surface");

		std::println("{}Surface created.{}", CLR::GRN, CLR::RESET);
	}

	// Pick a GPU and get the queues
	void pick_gpu_and_queues(vulkan_context &ctx, vkb::Instance &vkb_inst)
	{
		// Features from Vulkan 1.3
		auto features1_3 = vk::PhysicalDeviceVulkan13Features{
			.synchronization2 = true,
			.dynamicRendering = true,
		};

		// Features from Vulkan 1.2
		auto features1_2 = vk::PhysicalDeviceVulkan12Features{
			.descriptorIndexing                                 = true,
			.descriptorBindingUniformBufferUpdateAfterBind      = true,
			.descriptorBindingSampledImageUpdateAfterBind       = true,
			.descriptorBindingStorageImageUpdateAfterBind       = true,
			.descriptorBindingStorageBufferUpdateAfterBind      = true,
			.descriptorBindingUniformTexelBufferUpdateAfterBind = true,
			.descriptorBindingStorageTexelBufferUpdateAfterBind = true,
			.descriptorBindingPartiallyBound                    = true,
			.descriptorBindingVariableDescriptorCount           = true,
			.runtimeDescriptorArray                             = true,
			.timelineSemaphore                                  = true,
			.bufferDeviceAddress                                = true,
		};

		// Features from Vulkan 1.0/1
		auto features = vk::PhysicalDeviceFeatures{
			.samplerAnisotropy = true,
			.shaderInt64       = true,
		};

		// The descriptor buffer extension features
		auto descriptor_buffer_feature = vk::PhysicalDeviceDescriptorBufferFeaturesEXT{
			.descriptorBuffer                = true,
			.descriptorBufferPushDescriptors = true,
		};

		auto phy_dev_selector = vkb::PhysicalDeviceSelector{ vkb_inst };
		auto phy_dev_select   = phy_dev_selector
		                          .set_minimum_version(1, 3)
		                          .set_required_features_13(features1_3)
		                          .set_required_features_12(features1_2)
		                          .set_required_features(features)
		                          // Because descriptor buffer is not part of Core Vulkan 1.3, for some reason both
		                          // required_extension_name and required_extension_features are needed
		                          .add_required_extension(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)
		                          .add_required_extension_features(descriptor_buffer_feature)
		                          .set_surface(ctx.surface)
		                          .select();
		if (not phy_dev_select.has_value())
			std::println("Return value: {}", phy_dev_select.error().message());
		assert(phy_dev_select.has_value() == true and "Failed to select Physical GPU");

		// Enable the descriptor buffer extension features
		auto phy_dev_ret = phy_dev_select.value();
		auto res         = phy_dev_ret.enable_extension_features_if_present(descriptor_buffer_feature);
		assert(res == true and "Failed to enable extension features on GPU");

		// Create the Vulkan Device, this is logical device based on physical device
		auto device_builder = vkb::DeviceBuilder{ phy_dev_ret };
		auto vkb_device     = device_builder.build().value();

		// unwrap from vkb and wrap into our context
		ctx.chosen_gpu = phy_dev_ret.physical_device;
		ctx.device     = vkb_device.device;

		// Initialize the Dynamic Dispatch Loader using this device
		VULKAN_HPP_DEFAULT_DISPATCHER.init(ctx.device);

		// Get different queues from the device
		ctx.gfx_queue.queue  = vkb_device.get_queue(vkb::QueueType::graphics).value();
		ctx.gfx_queue.family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

		// Present queue is usually same as Graphics queue, for this example only graphics queue is used
		ctx.present_queue.queue  = vkb_device.get_queue(vkb::QueueType::present).value();
		ctx.present_queue.family = vkb_device.get_queue_index(vkb::QueueType::present).value();

		// Transfer queue is not used in this example
		ctx.transfer_queue.queue  = vkb_device.get_queue(vkb::QueueType::transfer).value();
		ctx.transfer_queue.family = vkb_device.get_queue_index(vkb::QueueType::transfer).value();

		// Compute queue is not used in this example
		ctx.compute_queue.queue  = vkb_device.get_queue(vkb::QueueType::compute).value();
		ctx.compute_queue.family = vkb_device.get_queue_index(vkb::QueueType::compute).value();

		auto gpu_props = ctx.chosen_gpu.getProperties();
		std::println("{}Selected GPU: {}{}",
		             CLR::GRN, gpu_props.deviceName.data(), CLR::RESET);

		std::println("{}Queues:{} Graphics: {}\n"
		             "\tPresent: {}\n"
		             "\tTransfer: {}\n"
		             "\tCompute: {}",
		             CLR::GRN, CLR::RESET,
		             ctx.gfx_queue.family,
		             ctx.present_queue.family,
		             ctx.transfer_queue.family,
		             ctx.compute_queue.family);
	}

	// Create a GPU Memory Allocator using Vulkan Memory Allocator
	void create_gpu_mem_allocator(vulkan_context &ctx)
	{
		auto allocator_info = vma::AllocatorCreateInfo{
			.flags          = vma::AllocatorCreateFlagBits::eBufferDeviceAddress,
			.physicalDevice = ctx.chosen_gpu,
			.device         = ctx.device,
			.instance       = ctx.instance,
		};

		ctx.mem_allocator = vma::createAllocator(allocator_info);

		std::println("{}GPU Memory Allocator created.{}", CLR::GRN, CLR::RESET);
	}

	// Create a Swapchain using VK-Bootstrap
	void create_swapchain(vulkan_context &ctx)
	{
		auto sfc_prop = ctx.chosen_gpu.getSurfaceCapabilitiesKHR(ctx.surface);
		auto width    = sfc_prop.currentExtent.width;
		auto height   = sfc_prop.currentExtent.height;

		// When debugging, enable VSync, else no VSync
		auto present_mode = (use_vulkan_validation_layers) ? vk::PresentModeKHR::eFifo : vk::PresentModeKHR::eImmediate;

		auto sc_builder = vkb::SwapchainBuilder{ ctx.chosen_gpu, ctx.device, ctx.surface };
		auto vkb_sc     = sc_builder
		                  .set_desired_format({
							.format     = VK_FORMAT_B8G8R8A8_UNORM,
							.colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
						  })
		                  .set_desired_extent(width, height)
		                  .set_desired_present_mode(static_cast<VkPresentModeKHR>(present_mode)) // Cast from C++ to C
		                  .build()
		                  .value();

		ctx.swap_chain = vkb_sc.swapchain;
		ctx.sc_extent  = vkb_sc.extent;
		ctx.sc_format  = static_cast<vk::Format>(vkb_sc.image_format); // convert from enum to enum-class

		// Convert from vector of VkImage to vector of vk::Image
		std::ranges::transform(
			vkb_sc.get_images().value(),
			std::back_inserter(ctx.sc_images),
			[](auto &&img) {
			return vk::Image(img);
		});

		// Convert from vector of VkImageView to vector of vk::ImageView
		std::ranges::transform(
			vkb_sc.get_image_views().value(),
			std::back_inserter(ctx.sc_views),
			[](auto &&img_vw) {
			return vk::ImageView(img_vw);
		});

		ctx.max_frame_count = static_cast<uint32_t>(ctx.sc_views.size());
		ctx.current_frame   = 0;

		std::println("{}Swapchain created.{}\n"
		             "\tImage size: {} x {}\n"
		             "\tImage count: {}",
		             CLR::GRN, CLR::RESET, width, height, ctx.max_frame_count);
	}

	// Create synchronization objects for each swapchain image
	void create_sync_objects(vulkan_context &ctx)
	{
		ctx.image_signals.resize(ctx.max_frame_count);
		for (auto &&[available, rendered, in_flight] : ctx.image_signals)
		{
			auto semaphore_info = vk::SemaphoreCreateInfo{};
			available           = ctx.device.createSemaphore(semaphore_info);
			rendered            = ctx.device.createSemaphore(semaphore_info);

			auto fence_info = vk::FenceCreateInfo{
				.flags = vk::FenceCreateFlagBits::eSignaled,
			};
			in_flight = ctx.device.createFence(fence_info);
		}

		auto tfr_fence_info = vk::FenceCreateInfo{
			.flags = vk::FenceCreateFlagBits::eSignaled,
		};
		ctx.tfr_in_flight_fence = ctx.device.createFence(tfr_fence_info);

		std::println("{}Semaphores and Fences created.{}",
		             CLR::GRN, CLR::RESET);
	}

	// Create a Command Pool and allocate Command Buffers for Graphics use
	void create_command_pool(vulkan_context &ctx)
	{
		// Create Graphics Command Pool
		auto command_pool_info = vk::CommandPoolCreateInfo{
			.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = ctx.gfx_queue.family,
		};
		ctx.gfx_command_pool = ctx.device.createCommandPool(command_pool_info);

		std::println("{}Graphics Command Pool created.{}", CLR::GRN, CLR::RESET);

		// Create Graphics Command Buffer
		auto command_buffer_alloc_info = vk::CommandBufferAllocateInfo{
			.commandPool        = ctx.gfx_command_pool,
			.level              = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = ctx.max_frame_count,
		};
		ctx.gfx_command_buffers = ctx.device.allocateCommandBuffers(command_buffer_alloc_info);

		std::println("{}Graphics Command Buffers allocated.{}", CLR::GRN, CLR::RESET);

		// Create Transfer Command Pool
		// command_pool_info.queueFamilyIndex = ctx.gfx_queue.family;
		command_pool_info.queueFamilyIndex = ctx.transfer_queue.family;
		ctx.tfr_command_pool               = ctx.device.createCommandPool(command_pool_info);
		std::println("{}Transfer Command Pool created.{}", CLR::GRN, CLR::RESET);

		// Create Tranfer Command Buffer
		command_buffer_alloc_info = vk::CommandBufferAllocateInfo{
			.commandPool        = ctx.tfr_command_pool,
			.level              = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = 1,
		};
		auto tfr_cbs           = ctx.device.allocateCommandBuffers(command_buffer_alloc_info);
		ctx.tfr_command_buffer = tfr_cbs.front();

		std::println("{}Transfer Command Buffer allocated.{}", CLR::GRN, CLR::RESET);
	}

	// Initialize Vulkan
	auto init_vulkan(GLFWwindow *window) -> vulkan_context
	{
		std::println("{}Intializing Vulkan...{}", CLR::BLU, CLR::RESET);

		auto vk_ctx   = vulkan_context{};
		auto vkb_inst = create_instance(vk_ctx);
		create_surface(vk_ctx, window);
		pick_gpu_and_queues(vk_ctx, vkb_inst);
		create_gpu_mem_allocator(vk_ctx);
		create_swapchain(vk_ctx);
		create_sync_objects(vk_ctx);
		create_command_pool(vk_ctx);

		return vk_ctx;
	}

	// Clean up Vulkan Objects
	void shutdown_vulkan(vulkan_context &ctx)
	{
		std::println("{}Shutting down Vulkan...{}", CLR::BLU, CLR::RESET);

		ctx.device.waitIdle();

		// Destroy Command Pool and Buffers
		ctx.tfr_command_buffer = nullptr;
		ctx.device.destroyCommandPool(ctx.tfr_command_pool);

		ctx.gfx_command_buffers.clear();
		ctx.device.destroyCommandPool(ctx.gfx_command_pool);

		// Destroy Semaphores and Fences
		ctx.device.destroyFence(ctx.tfr_in_flight_fence);
		for (auto &&[available, rendered, in_flight] : ctx.image_signals)
		{
			ctx.device.destroyFence(in_flight);
			ctx.device.destroySemaphore(rendered);
			ctx.device.destroySemaphore(available);
		}
		ctx.image_signals.clear();

		// Destroy Swapchain
		std::ranges::for_each(ctx.sc_views, [&](auto &&img_vw) {
			ctx.device.destroyImageView(img_vw);
		});
		ctx.sc_views.clear();
		ctx.sc_images.clear();
		ctx.device.destroySwapchainKHR(ctx.swap_chain);

		// Destroy GPU Memory Allocator
		ctx.mem_allocator.destroy();

		// Destroy Surface
		ctx.instance.destroySurfaceKHR(ctx.surface);

		// Destroy Debug Messenger and Instance
		ctx.device.destroy();
		vkb::destroy_debug_utils_messenger(ctx.instance, ctx.debug_messenger);
		ctx.instance.destroy();
	}
}

/*
 * Vulkan Objects used every frame
 */
namespace frame
{
	// Maximum time to wait for fence
	constexpr auto wait_time = UINT_MAX;

	// Basic GPU Memory Buffer structure
	struct gpu_buffer
	{
		vma::Allocation allocation;
		vma::AllocationInfo info;
		vk::Buffer buffer;
		vk::DeviceSize address;
		vk::DeviceSize size;
	};

	// GPU Image structure
	struct gpu_image
	{
		vma::Allocation allocation;
		vk::Image image;
		vk::ImageView view;
		vk::Extent3D extent;
		vk::Format format;
		vk::ImageAspectFlags aspect_mask;
		uint32_t mipmap_levels;
	};

	// GPU Descriptor Set Buffer
	struct gpu_descriptor
	{
		vk::DescriptorSetLayout layout;
		vk::DeviceSize layout_size;
		vk::DeviceSize layout_offset;
		gpu_buffer buffer;
	};

	// This holds all the data required to render a frame
	struct render_context
	{
		// Created by create_pipeline
		vk::PipelineLayout layout;
		vk::Pipeline pipeline;

		// Populated by main function
		std::array<float, 4> clear_color;

		// Created by create_uniform_descriptor_set
		gpu_descriptor uniform_descriptor;
		vk::PhysicalDeviceDescriptorBufferPropertiesEXT descriptor_buffer_props;

		// Created by create_image_descriptor_set
		gpu_descriptor texture_descriptor;

		// Created by create_uniform_buffer
		std::vector<gpu_buffer> uniform_buffers;

		// Created by create_texture_buffer
		gpu_buffer texture_buffer;

		// Created by create_texture_image
		gpu_image texture_image;

		// Created by create_texture_sampler
		vk::Sampler texture_sampler;
	};

	// Structure to hold shader binary data, for vertex and fragment shaders
	struct shader_binaries
	{
		std::vector<std::byte> vertex{};
		std::vector<std::byte> fragment{};
	};

	// Check if the format is a depth only format
	auto is_depth_only_format(vk::Format format) -> bool
	{
		return format == vk::Format::eD16Unorm ||
		       format == vk::Format::eD32Sfloat;
	}

	// Check if the format is a depth stencil format
	// Not used in this example
	auto is_depth_stencil_format(vk::Format format) -> bool
	{
		return format == vk::Format::eD16UnormS8Uint ||
		       format == vk::Format::eD24UnormS8Uint ||
		       format == vk::Format::eD32SfloatS8Uint;
	}

	// Create a graphics pipeline and layout
	void create_pipeline(const base::vulkan_context &ctx,
	                     render_context &rndr,
	                     const shader_binaries &shaders,
	                     vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList,
	                     vk::PolygonMode polygon_mode   = vk::PolygonMode::eFill,
	                     vk::CullModeFlags cull_mode    = vk::CullModeFlagBits::eFront,
	                     vk::FrontFace front_face       = vk::FrontFace::eCounterClockwise,
	                     vk::Format depth_format        = vk::Format::eUndefined)
	{
		// Lambda to create vk::ShaderModule
		auto create_shader_module = [&](const std::span<const std::byte> shader_bin) -> vk::ShaderModule {
			auto shader_info = vk::ShaderModuleCreateInfo{
				.codeSize = shader_bin.size(),
				.pCode    = reinterpret_cast<const uint32_t *>(shader_bin.data()),
			};

			return ctx.device.createShaderModule(shader_info);
		};

		// Assume shaders will always have vertex and fragment shaders

		// Convert shader binary into shader modules
		using shader_stage_module = std::tuple<vk::ShaderStageFlagBits, vk::ShaderModule>;
		auto shader_list          = std::vector<shader_stage_module>{
            { vk::ShaderStageFlagBits::eVertex, create_shader_module(shaders.vertex) },
            { vk::ShaderStageFlagBits::eFragment, create_shader_module(shaders.fragment) },
		};

		// Shader Stages
		// Assume all shaders will have main function as entry point
		auto shader_stage_infos = std::vector<vk::PipelineShaderStageCreateInfo>{};
		std::ranges::transform(
			shader_list,
			std::back_inserter(shader_stage_infos),
			[](const shader_stage_module &stg_module) {
			return vk::PipelineShaderStageCreateInfo{
				.stage  = std::get<vk::ShaderStageFlagBits>(stg_module),
				.module = std::get<vk::ShaderModule>(stg_module),
				.pName  = "main" // Assume all shaders will have main function as entry point
			};
		});

		// Empty VertexInputStateCreateInfo, as system won't be using it
		auto vertex_input_info = vk::PipelineVertexInputStateCreateInfo{};

		// Input Assembly
		auto input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo{
			.topology               = topology,
			.primitiveRestartEnable = false,
		};

		// Viewport
		auto viewport_info = vk::PipelineViewportStateCreateInfo{
			.viewportCount = 1,
			.scissorCount  = 1,
		};

		// Rasterization
		auto rasterization_info = vk::PipelineRasterizationStateCreateInfo{
			.depthClampEnable        = false,
			.rasterizerDiscardEnable = false,
			.polygonMode             = polygon_mode,
			.cullMode                = cull_mode,
			.frontFace               = front_face,
			.depthBiasEnable         = false,
			.lineWidth               = 1.0f,
		};

		// Multisample anti-aliasing
		auto multisample_info = vk::PipelineMultisampleStateCreateInfo{
			.rasterizationSamples = vk::SampleCountFlagBits::e1, // should this be higher for higher msaa?
			.sampleShadingEnable  = false,
		};

		// Color Blend Attachment
		auto color_blend_attach_st = vk::PipelineColorBlendAttachmentState{
			.blendEnable    = false,
			.colorWriteMask = vk::ColorComponentFlagBits::eR |
			                  vk::ColorComponentFlagBits::eG |
			                  vk::ColorComponentFlagBits::eB |
			                  vk::ColorComponentFlagBits::eA,
		};

		// Color Blend State
		auto color_blend_info = vk::PipelineColorBlendStateCreateInfo{
			.logicOpEnable   = false,
			.logicOp         = vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments    = &color_blend_attach_st,
			.blendConstants  = std::array{ 0.f, 0.f, 0.f, 0.f },
		};

		// Dynamic States
		auto dynamic_states = std::vector{
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor,
		};

		auto dynamic_state_info = vk::PipelineDynamicStateCreateInfo{
			.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
			.pDynamicStates    = dynamic_states.data(),
		};

		// Descriptor Sets used by shaders
		auto descriptor_set_layouts = std::array{
			rndr.uniform_descriptor.layout, // UBO Set 0
			rndr.uniform_descriptor.layout, // UBO Set 1
			rndr.texture_descriptor.layout, // Texture Set 2
		};

		// Pipeline Layout with descriptor set layouts
		auto pipeline_layout_info = vk::PipelineLayoutCreateInfo{
			.setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size()),
			.pSetLayouts    = descriptor_set_layouts.data(),
		};
		rndr.layout = ctx.device.createPipelineLayout(pipeline_layout_info);

		// array of color formats, only one for now
		auto color_formats = std::array{ ctx.sc_format };

		// Pipeline Rendering Info
		auto pipeline_rendering_info = vk::PipelineRenderingCreateInfo{
			.colorAttachmentCount    = static_cast<uint32_t>(color_formats.size()),
			.pColorAttachmentFormats = color_formats.data(),
			.depthAttachmentFormat   = depth_format,
		};
		if (not is_depth_only_format(depth_format))
		{
			pipeline_rendering_info.stencilAttachmentFormat = depth_format;
		}

		// Finally create Pipeline
		auto pipeline_info = vk::GraphicsPipelineCreateInfo{
			.pNext               = &pipeline_rendering_info,
			.flags               = vk::PipelineCreateFlagBits::eDescriptorBufferEXT, // for descriptor buffer
			.stageCount          = static_cast<uint32_t>(shader_stage_infos.size()),
			.pStages             = shader_stage_infos.data(),
			.pVertexInputState   = &vertex_input_info,
			.pInputAssemblyState = &input_assembly_info,
			.pViewportState      = &viewport_info,
			.pRasterizationState = &rasterization_info,
			.pMultisampleState   = &multisample_info,
			.pColorBlendState    = &color_blend_info,
			.pDynamicState       = &dynamic_state_info,
			.layout              = rndr.layout,
			.subpass             = 0,
		};

		auto result_value = ctx.device.createGraphicsPipeline(nullptr, pipeline_info);
		assert(result_value.result == vk::Result::eSuccess and "Failed to create Graphics Pipeline");

		rndr.pipeline = result_value.value;

		// Destroy the shader modules
		for (auto &&[stg, mod] : shader_list)
		{
			ctx.device.destroyShaderModule(mod);
		}

		std::println("{}Pipeline created.{}", CLR::CYN, CLR::RESET);
	}

	// Structure to make image_layout_transition easier to use
	struct image_transition_info
	{
		vk::PipelineStageFlags2 src_stage_mask;
		vk::PipelineStageFlags2 dst_stage_mask;
		vk::AccessFlags2 src_access_mask;
		vk::AccessFlags2 dst_access_mask;
		vk::ImageLayout old_layout;
		vk::ImageLayout new_layout;
		vk::ImageSubresourceRange subresource_range;
	};

	// Helper functions to get vk::PipelineStageFlags and vk::AccessFlags from vk::ImageLayout
	auto get_pipeline_stage_flags(vk::ImageLayout image_layout) -> vk::PipelineStageFlags2
	{
		switch (image_layout)
		{
			using il = vk::ImageLayout;
			using pf = vk::PipelineStageFlagBits2;

		case il::eUndefined:
			return pf::eTopOfPipe;
		case il::ePreinitialized:
			return pf::eHost;
		case il::eTransferSrcOptimal:
		case il::eTransferDstOptimal:
			return pf::eTransfer;
		case il::eColorAttachmentOptimal:
			return pf::eColorAttachmentOutput;
		case il::eDepthAttachmentOptimal:
			return pf::eEarlyFragmentTests | pf::eLateFragmentTests;
		case il::eFragmentShadingRateAttachmentOptimalKHR:
			return pf::eFragmentShadingRateAttachmentKHR;
		case il::eReadOnlyOptimal:
		case il::eShaderReadOnlyOptimal:
			return pf::eVertexShader | pf::eFragmentShader;
		case il::ePresentSrcKHR:
			return pf::eBottomOfPipe;
		case il::eGeneral:
			assert(false and "Don't know how to get a meaningful vk::PipelineStageFlags for VK_IMAGE_LAYOUT_GENERAL! Don't use it!");
		default:
			assert(false and "Unknown layout flag");
		}
		return {};
	}

	auto get_access_flags(vk::ImageLayout image_layout) -> vk::AccessFlags2
	{
		switch (image_layout)
		{
			using il = vk::ImageLayout;
			using af = vk::AccessFlagBits2;

		case il::eUndefined:
		case il::ePresentSrcKHR:
			return af::eNone;
		case il::ePreinitialized:
			return af::eHostWrite;
		case il::eColorAttachmentOptimal:
			return af::eColorAttachmentRead | af::eColorAttachmentWrite;
		case il::eDepthAttachmentOptimal:
			return af::eDepthStencilAttachmentRead | af::eDepthStencilAttachmentWrite;
		case il::eFragmentShadingRateAttachmentOptimalKHR:
			return af::eFragmentShadingRateAttachmentReadKHR;
		case il::eShaderReadOnlyOptimal:
			return af::eShaderRead | af::eInputAttachmentRead;
		case il::eTransferSrcOptimal:
			return af::eTransferRead;
		case il::eTransferDstOptimal:
			return af::eTransferWrite;
		case il::eGeneral:
			assert(false and "Don't know how to get a meaningful vk::AccessFlags for VK_IMAGE_LAYOUT_GENERAL! Don't use it!");
		default:
			assert(false and "Unknown layout flag");
		}
		return {};
	}

	// Add Pipeline Barrier to transition image from old_layout to new_layout, used by update_command_buffer
	void image_layout_transition(vk::CommandBuffer cb, vk::Image image, const image_transition_info &iti)
	{
		auto image_memory_barrier = vk::ImageMemoryBarrier2{
			.srcStageMask        = iti.src_stage_mask,
			.srcAccessMask       = iti.src_access_mask,
			.dstStageMask        = iti.dst_stage_mask,
			.dstAccessMask       = iti.dst_access_mask,
			.oldLayout           = iti.old_layout,
			.newLayout           = iti.new_layout,
			.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
			.dstQueueFamilyIndex = vk::QueueFamilyIgnored,
			.image               = image,
			.subresourceRange    = iti.subresource_range,
		};

		auto dep_info = vk::DependencyInfo{
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers    = &image_memory_barrier,
		};

		cb.pipelineBarrier2(dep_info);
	}

	// overload image_layout_transition with fewer parameters, used by update_command_buffer
	void image_layout_transition(vk::CommandBuffer cb,
	                             vk::Image image,
	                             vk::ImageLayout old_layout, vk::ImageLayout new_layout,
	                             const vk::ImageSubresourceRange &subresource_range)
	{
		image_layout_transition(
			cb,
			image,
			image_transition_info{
			  .src_stage_mask    = get_pipeline_stage_flags(old_layout),
			  .dst_stage_mask    = get_pipeline_stage_flags(new_layout),
			  .src_access_mask   = get_access_flags(old_layout),
			  .dst_access_mask   = get_access_flags(new_layout),
			  .old_layout        = old_layout,
			  .new_layout        = new_layout,
			  .subresource_range = subresource_range,
			});
	}

	// overload with even fewer parameters, this one is used by Buffer to Image copy
	void image_layout_transition(vk::CommandBuffer cb,
	                             vk::Image image,
	                             vk::ImageLayout old_layout, vk::ImageLayout new_layout)
	{
		using pf = vk::PipelineStageFlagBits2;
		using af = vk::AccessFlagBits2;

		auto aspect_mask = (new_layout == vk::ImageLayout::eDepthAttachmentOptimal)
		                     ? vk::ImageAspectFlagBits::eDepth
		                     : vk::ImageAspectFlagBits::eColor;

		auto sub_res_rng = vk::ImageSubresourceRange{
			.aspectMask     = aspect_mask,
			.baseMipLevel   = 0,
			.levelCount     = vk::RemainingMipLevels,
			.baseArrayLayer = 0,
			.layerCount     = vk::RemainingArrayLayers,
		};

		image_layout_transition(
			cb,
			image,
			image_transition_info{
			  .src_stage_mask    = pf::eAllCommands,
			  .dst_stage_mask    = pf::eAllCommands,
			  .src_access_mask   = af::eMemoryWrite,
			  .dst_access_mask   = af::eMemoryWrite | af::eMemoryRead,
			  .old_layout        = old_layout,
			  .new_layout        = new_layout,
			  .subresource_range = sub_res_rng,
			});
	}

	// return value such that 'size' is adjusted to match the memory 'alignment' requirements of the device
	auto align_size(vk::DeviceSize size, vk::DeviceSize alignment) -> vk::DeviceSize
	{
		return (size + alignment - 1) & ~(alignment - 1);
	}

	// Create Descriptor Set
	void create_uniform_descriptor_set(const base::vulkan_context &ctx, render_context &rndr)
	{
		// Get descriptor buffer properties
		auto device_props = vk::PhysicalDeviceProperties2{
			.pNext = &rndr.descriptor_buffer_props,
		};
		ctx.chosen_gpu.getProperties2(&device_props);

		auto &descriptor = rndr.uniform_descriptor;

		constexpr auto binding = 0u; // Binding number for the descriptor set, i.e register(b0, space#) in HLSL

		// Descriptor Set Layout for Uniform Buffer
		auto desc_set_layout_binding = vk::DescriptorSetLayoutBinding{
			.binding         = binding,
			.descriptorType  = vk::DescriptorType::eUniformBuffer, // What kind of descriptor set is this?
			.descriptorCount = 1,
			.stageFlags      = vk::ShaderStageFlagBits::eVertex, // Where will it be used by shaders?
		};

		auto desc_set_layout_info = vk::DescriptorSetLayoutCreateInfo{
			.flags        = vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT,
			.bindingCount = 1,
			.pBindings    = &desc_set_layout_binding,
		};

		// Create the set_layout object for Uniform Buffer
		descriptor.layout = ctx.device.createDescriptorSetLayout(desc_set_layout_info);
		// Size of descriptor set layout for Uniform Buffer
		descriptor.layout_size = ctx.device.getDescriptorSetLayoutSizeEXT(descriptor.layout);
		// Offset of descriptor set layout, based on binding value, I think. for Uniform Buffer
		descriptor.layout_offset = ctx.device.getDescriptorSetLayoutBindingOffsetEXT(descriptor.layout, binding);

		// Align the size to the required alignment
		descriptor.layout_size = align_size(descriptor.layout_size, rndr.descriptor_buffer_props.descriptorBufferOffsetAlignment);

		std::println("{}Uniform buffer descriptor set created.{}", CLR::CYN, CLR::RESET);
	}

	// Create Image and Sampler Descriptor Set
	void create_texture_descriptor_set(const base::vulkan_context &ctx, render_context &rndr)
	{
		auto &descriptor = rndr.texture_descriptor;

		constexpr auto binding = 0u; // Binding number for the descriptor set, t0, s0

		// Descriptor Set Layout for Image & Sampler
		// Rest is mostly same as Uniform Descriptor Set
		auto desc_set_layout_binding = vk::DescriptorSetLayoutBinding{
			.binding         = binding,
			.descriptorType  = vk::DescriptorType::eCombinedImageSampler, // Instead of UniformBuffer bit, we want Image+Sampler
			.descriptorCount = 1,
			.stageFlags      = vk::ShaderStageFlagBits::eFragment, // Instead of vertex, this is for Fragment/Pixel shader
		};

		auto desc_set_layout_info = vk::DescriptorSetLayoutCreateInfo{
			.flags        = vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT,
			.bindingCount = 1,
			.pBindings    = &desc_set_layout_binding,
		};

		descriptor.layout        = ctx.device.createDescriptorSetLayout(desc_set_layout_info);
		descriptor.layout_size   = ctx.device.getDescriptorSetLayoutSizeEXT(descriptor.layout);
		descriptor.layout_offset = ctx.device.getDescriptorSetLayoutBindingOffsetEXT(descriptor.layout, binding);

		descriptor.layout_size = align_size(descriptor.layout_size, rndr.descriptor_buffer_props.descriptorBufferOffsetAlignment);

		std::println("{}Texture Descriptor set created{}", CLR::CYN, CLR::RESET);
	}

	// Create Descriptor Buffer and allocation
	void create_uniform_descriptor_buffer(const base::vulkan_context &ctx, render_context &rndr)
	{
		auto &udb = rndr.uniform_descriptor.buffer;

		auto buffer_info = vk::BufferCreateInfo{
			.size  = rndr.uniform_descriptor.layout_size * 2,              // There will be two uniform descriptor sets Projection and Transforms
			.usage = vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT // This is a descriptor buffer
			       | vk::BufferUsageFlagBits::eShaderDeviceAddress,        // Need to be able to get GPU memory address
		};

		auto alloc_info = vma::AllocationCreateInfo{
			.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped,
			.usage = vma::MemoryUsage::eAuto, // let VMA figure out what's optimal place.
		};

		// This line inits vk::Buffer, vma::Allocation, and vma::AllocationInfo
		std::tie(udb.buffer, udb.allocation) = ctx.mem_allocator.createBuffer(buffer_info, alloc_info, &udb.info);

		auto buff_addr_info = vk::BufferDeviceAddressInfo{
			.buffer = udb.buffer,
		};
		udb.address = ctx.device.getBufferAddress(buff_addr_info); // Get GPU address
		udb.size    = udb.info.size;                               // This is probably unnecessary, seems to just duplicate AllocationInfo

		// Give the buffer a name for debugging
		ctx.device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
		  .objectType   = vk::ObjectType::eBuffer,
		  .objectHandle = (uint64_t)(static_cast<VkBuffer>(udb.buffer)),
		  .pObjectName  = "Uniform Descriptor Buffer",
		});

		std::println("{}Uniform Descriptor Buffer created.{}", CLR::CYN, CLR::RESET);
	}

	void create_texture_descriptor_buffer(const base::vulkan_context &ctx, render_context &rndr)
	{
		auto &tdb = rndr.texture_descriptor.buffer;

		auto buffer_info = vk::BufferCreateInfo{
			.size  = rndr.texture_descriptor.layout_size,                  // There is only one descriptor set for texture
			.usage = vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT // This is a descriptor buffer
			       | vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT  // This descriptor is also has sampler
			       | vk::BufferUsageFlagBits::eShaderDeviceAddress,        // Need to be able to get GPU memory address
		};

		auto alloc_info = vma::AllocationCreateInfo{
			.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped,
			.usage = vma::MemoryUsage::eAuto, // let VMA figure out what's optimal place.
		};

		// This line inits vk::Buffer, vma::Allocation, and vma::AllocationInfo
		std::tie(tdb.buffer, tdb.allocation) = ctx.mem_allocator.createBuffer(buffer_info, alloc_info, &tdb.info);

		auto buff_addr_info = vk::BufferDeviceAddressInfo{
			.buffer = tdb.buffer,
		};
		tdb.address = ctx.device.getBufferAddress(buff_addr_info); // Get GPU address
		tdb.size    = tdb.info.size;                               // This is probably unnecessary, seems to just duplicate AllocationInfo

		// Give the buffer a name for debugging
		ctx.device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
		  .objectType   = vk::ObjectType::eBuffer,
		  .objectHandle = (uint64_t)(static_cast<VkBuffer>(tdb.buffer)),
		  .pObjectName  = "Texture Descriptor Buffer",
		});

		std::println("{}Texture Descriptor Buffer created.{}", CLR::CYN, CLR::RESET);
	}

	void create_uniform_buffer(const base::vulkan_context &ctx, render_context &rndr, std::span<uint32_t> sizes)
	{
		rndr.uniform_buffers.resize(sizes.size());
		auto idx = 0u;

		for (auto &&[ubo, size] : std::views::zip(rndr.uniform_buffers, sizes))
		{
			auto buffer_info = vk::BufferCreateInfo{
				.size  = size,
				.usage = vk::BufferUsageFlagBits::eUniformBuffer        // So far, we only have Uniform Buffers
				       | vk::BufferUsageFlagBits::eShaderDeviceAddress, // We want to be able to get GPU side memory address for this object
			};

			auto alloc_info = vma::AllocationCreateInfo{
				.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped,
				.usage = vma::MemoryUsage::eAuto, // let VMA figure out what's optimal place.
			};

			// This line inits vk::Buffer, vma::Allocation, and vma::AllocationInfo
			std::tie(ubo.buffer, ubo.allocation) = ctx.mem_allocator.createBuffer(buffer_info, alloc_info, &ubo.info);

			auto buff_addr_info = vk::BufferDeviceAddressInfo{
				.buffer = ubo.buffer,
			};
			ubo.address = ctx.device.getBufferAddress(buff_addr_info); // Get GPU address
			ubo.size    = ubo.info.size;                               // This is probably unnecessary, seems to just duplicate AllocationInfo

			// Give the buffer a name for debugging
			ctx.device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
			  .objectType   = vk::ObjectType::eBuffer,
			  .objectHandle = (uint64_t)(static_cast<VkBuffer>(ubo.buffer)),
			  .pObjectName  = std::format("Uniform Buffer {}", idx++).c_str(),
			});
		}

		std::println("{}Uniform Buffers created [{}].{}", CLR::CYN, idx, CLR::RESET);
	}

	// Populate Descriptor Buffer with Uniform Buffer Address and Size
	void populate_uniform_descriptor_buffer(const base::vulkan_context &ctx, render_context &rndr)
	{
		auto &uds  = rndr.uniform_descriptor;
		auto &dsbo = uds.buffer;

		// Get pointer to CPU-side memory location
		auto descriptor_buffer_ptr = ctx.mem_allocator.mapMemory(dsbo.allocation);

		for (auto &&[idx, ubo] : rndr.uniform_buffers | std::views::enumerate)
		{
			auto address_info = vk::DescriptorAddressInfoEXT{
				.address = ubo.address, // UBO device address
				.range   = ubo.size,    // UBO device size
				.format  = vk::Format::eUndefined,
			};

			auto buff_descriptor_info = vk::DescriptorGetInfoEXT{
				.type = vk::DescriptorType::eUniformBuffer, // So far we only have Uniform Buffers
				.data = {
				  .pUniformBuffer = &address_info,
				},
			};

			auto offset   = idx * (uds.layout_offset + uds.layout_size);   // Offset for each descriptor set for each UBO
			auto buff_ptr = io::offset_ptr(descriptor_buffer_ptr, offset); // Get Offset Descriptor pointer

			// Write to descriptor buffer, the UBO location and size.
			ctx.device.getDescriptorEXT(&buff_descriptor_info, rndr.descriptor_buffer_props.uniformBufferDescriptorSize, buff_ptr);
		}

		// Send data to GPU
		ctx.mem_allocator.unmapMemory(dsbo.allocation); // Unmap; or flush to keep mapped object alive on CPU side.

		std::println("{}Uniform Descriptor Buffer populated.{}", CLR::CYN, CLR::RESET);
	}

	// Populate Descriptor Buffer with Uniform Buffer Address and Size
	void populate_texture_descriptor_buffer(const base::vulkan_context &ctx, render_context &rndr)
	{
		auto &tds  = rndr.texture_descriptor;
		auto &dsbo = tds.buffer;

		// Get pointer to CPU-side memory location
		auto descriptor_buffer_ptr = ctx.mem_allocator.mapMemory(dsbo.allocation);

		auto descriptor_img_info = vk::DescriptorImageInfo{
			.sampler     = rndr.texture_sampler,
			.imageView   = rndr.texture_image.view,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		};

		auto buff_descriptor_info = vk::DescriptorGetInfoEXT{
			.type = vk::DescriptorType::eCombinedImageSampler,
			.data = {
			  .pCombinedImageSampler = &descriptor_img_info,
			},
		};

		auto offset   = 0;                                             // no offset, as this is only texture descriptor
		auto buff_ptr = io::offset_ptr(descriptor_buffer_ptr, offset); // Get Offset Descriptor pointer

		// Write to descriptor buffer, the UBO location and size.
		ctx.device.getDescriptorEXT(&buff_descriptor_info, rndr.descriptor_buffer_props.combinedImageSamplerDescriptorSize, buff_ptr);

		// Send data to GPU
		ctx.mem_allocator.unmapMemory(dsbo.allocation); // Unmap; or flush to keep mapped object alive on CPU side.

		std::println("{}Texture Descriptor Buffer populated.{}", CLR::CYN, CLR::RESET);
	}

	// Populate Uniform Buffer with data, in this example Perspective Projection Matrix and Instance Transform Matrix
	void populate_uniform_buffer(const base::vulkan_context &ctx, render_context &rndr, io::byte_spans data)
	{
		for (auto &&[ubo, ubo_data] : std::views::zip(rndr.uniform_buffers, data))
		{
			auto ubo_ptr = ctx.mem_allocator.mapMemory(ubo.allocation);

			std::memcpy(ubo_ptr, ubo_data.data(), ubo_data.size());

			ctx.mem_allocator.unmapMemory(ubo.allocation); // Unmap; or flush to keep mapped object alive on CPU side.
		}

		std::println("{}Uniform Buffers populated.{}", CLR::CYN, CLR::RESET);
	}

	// Create texture staging buffer
	void create_texture_buffer(const base::vulkan_context &ctx, render_context &rndr, uint32_t tex_size)
	{
		auto &tex = rndr.texture_buffer;

		auto buffer_info = vk::BufferCreateInfo{
			.size  = tex_size,
			.usage = vk::BufferUsageFlagBits::eTransferSrc          // We will transfer this to vk::Image
			       | vk::BufferUsageFlagBits::eShaderDeviceAddress, // We want to be able to get GPU side memory address for this object
		};

		auto alloc_info = vma::AllocationCreateInfo{
			.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped,
			.usage = vma::MemoryUsage::eAuto, // let VMA figure out what's optimal place.
		};

		// This line inits vk::Buffer, vma::Allocation, and vma::AllocationInfo
		std::tie(tex.buffer, tex.allocation) = ctx.mem_allocator.createBuffer(buffer_info, alloc_info, &tex.info);

		auto buff_addr_info = vk::BufferDeviceAddressInfo{
			.buffer = tex.buffer,
		};
		tex.address = ctx.device.getBufferAddress(buff_addr_info); // Get GPU address
		tex.size    = tex.info.size;                               // This is probably unnecessary, seems to just duplicate AllocationInfo

		// Give the buffer a name for debugging
		ctx.device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
		  .objectType   = vk::ObjectType::eBuffer,
		  .objectHandle = (uint64_t)(static_cast<VkBuffer>(tex.buffer)),
		  .pObjectName  = "Texture Staging Buffer",
		});

		std::println("{}Texture Staging Buffer created.{}", CLR::CYN, CLR::RESET);
	}

	// Populate texture staging buffer
	void populate_texture_buffer(const base::vulkan_context &ctx, render_context &rndr, io::byte_span tex_data)
	{
		auto &tex = rndr.texture_buffer;

		auto tex_ptr = ctx.mem_allocator.mapMemory(tex.allocation);

		std::memcpy(tex_ptr, tex_data.data(), tex_data.size());

		ctx.mem_allocator.unmapMemory(tex.allocation);

		std::println("{}Texture Staging Buffer populated.{}", CLR::CYN, CLR::RESET);
	}

	auto ddsktxfmt_to_vkfmt(ddsktx_format fmt) -> vk::Format
	{
		switch (fmt)
		{
			using vf = vk::Format;
		case DDSKTX_FORMAT_BC1: // DXT1
			return vf::eBc1RgbUnormBlock;
		case DDSKTX_FORMAT_BC2: // DXT3
			return vf::eBc2UnormBlock;
		case DDSKTX_FORMAT_BC3: // DXT5
			return vf::eBc3UnormBlock;
		case DDSKTX_FORMAT_BC4: // ATI1
			return vf::eBc4UnormBlock;
		case DDSKTX_FORMAT_BC5: // ATI2
			return vf::eBc5UnormBlock;
		case DDSKTX_FORMAT_BC6H: // BC6H
			return vf::eBc6HSfloatBlock;
		case DDSKTX_FORMAT_BC7: // BC7
			return vf::eBc7UnormBlock;
		default:
			break;
		}

		assert(false and "Unmapped ddsktx_format, no conversion available");
		return vk::Format::eUndefined;
	}

	// Create vk::Image to hold Texture for shader
	void create_texture_image(const base::vulkan_context &ctx, render_context &rndr, ddsktx_texture_info image_hdr)
	{
		auto &img = rndr.texture_image;

		img.format = ddsktxfmt_to_vkfmt(image_hdr.format);
		img.extent = vk::Extent3D{
			.width  = static_cast<uint32_t>(image_hdr.width),
			.height = static_cast<uint32_t>(image_hdr.height),
			.depth  = static_cast<uint32_t>(image_hdr.depth),
		};
		img.aspect_mask   = vk::ImageAspectFlagBits::eColor;
		img.mipmap_levels = static_cast<uint32_t>(image_hdr.num_mips);

		// Create Image
		auto image_info = vk::ImageCreateInfo{
			.imageType   = vk::ImageType::e2D,
			.format      = img.format,
			.extent      = img.extent,
			.mipLevels   = img.mipmap_levels,
			.arrayLayers = static_cast<uint32_t>(image_hdr.num_layers),
			.usage       = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
		};

		auto alloc_info = vma::AllocationCreateInfo{
			.usage         = vma::MemoryUsage::eAuto,
			.requiredFlags = vk::MemoryPropertyFlagBits::eDeviceLocal
		};

		std::tie(img.image, img.allocation) = ctx.mem_allocator.createImage(image_info, alloc_info);

		// Create ImageView
		auto view_info = vk::ImageViewCreateInfo{
			.image      = img.image,
			.viewType   = vk::ImageViewType::e2D,
			.format     = img.format,
			.components = {
			  .r = vk::ComponentSwizzle::eIdentity,
			  .g = vk::ComponentSwizzle::eIdentity,
			  .b = vk::ComponentSwizzle::eIdentity,
			  .a = vk::ComponentSwizzle::eIdentity,
			},
			.subresourceRange = {
			  .aspectMask     = img.aspect_mask,
			  .baseMipLevel   = 0,
			  .levelCount     = static_cast<uint32_t>(image_hdr.num_mips),
			  .baseArrayLayer = 0,
			  .layerCount     = static_cast<uint32_t>(image_hdr.num_layers),
			},
		};

		img.view = ctx.device.createImageView(view_info);

		std::println("{}GPU Texture Image and View created.{}", CLR::CYN, CLR::RESET);
	}

	void create_texture_sampler(const base::vulkan_context &ctx, render_context &rndr)
	{
		// Enable anisotropic filtering
		// This feature is optional, so we must check if it's supported on the device
		float max_anisotropy = 1.0f;
		if (ctx.chosen_gpu.getFeatures().samplerAnisotropy)
		{
			// Use max. level of anisotropy for this example
			max_anisotropy = ctx.chosen_gpu.getProperties().limits.maxSamplerAnisotropy;
		}

		auto sampler_info = vk::SamplerCreateInfo{
			.magFilter        = vk::Filter::eLinear,
			.minFilter        = vk::Filter::eLinear,
			.mipmapMode       = vk::SamplerMipmapMode::eLinear,
			.addressModeU     = vk::SamplerAddressMode::eRepeat,
			.addressModeV     = vk::SamplerAddressMode::eRepeat,
			.addressModeW     = vk::SamplerAddressMode::eRepeat,
			.anisotropyEnable = true,
			.maxAnisotropy    = max_anisotropy,
			.compareEnable    = false,
			.minLod           = 0,
			.maxLod           = 0, // static_cast<float>(rndr.texture_image.mipmap_levels),
			.borderColor      = vk::BorderColor::eFloatOpaqueWhite,
		};

		rndr.texture_sampler = ctx.device.createSampler(sampler_info);

		std::println("{}Texture Sampler Created.{}", CLR::CYN, CLR::RESET);
	}

	void copy_texture_buffer_to_image(const base::vulkan_context &ctx, render_context &rndr, const std::span<const io::texture::sub_data> mips_info)
	{
		auto &cb  = ctx.tfr_command_buffer;
		auto &img = rndr.texture_image;
		auto &tb  = rndr.texture_buffer;

		// Reset Tranfer queue fence
		ctx.device.resetFences(ctx.tfr_in_flight_fence);

		// Reset transfer command buffer
		cb.reset();
		auto cb_begin_info = vk::CommandBufferBeginInfo{};
		// Begin recording new transfer commands
		auto cb_result = cb.begin(&cb_begin_info);
		assert(cb_result == vk::Result::eSuccess and "Failed to begin transfer command buffer");

		// auto sub_res_rng = vk::ImageSubresourceRange{
		// 	.aspectMask     = img.aspect_mask,
		// 	.baseMipLevel   = 0,
		// 	.levelCount     = img.mipmap_levels,
		// 	.baseArrayLayer = 0,
		// 	.layerCount     = 1,
		// };

		// image_layout_transition(cb, img.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, sub_res_rng);
		image_layout_transition(cb, img.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

		auto copy_regions = std::vector<vk::BufferImageCopy>{};
		std::ranges::transform(mips_info, std::back_inserter(copy_regions), [&](auto &info) {
			return vk::BufferImageCopy{
				.bufferOffset     = info.offset,
				.imageSubresource = {
				  .aspectMask     = img.aspect_mask,
				  .mipLevel       = info.mip_idx,
				  .baseArrayLayer = info.layer_idx,
				  .layerCount     = 1,
				},
				.imageExtent = {
				  .width  = info.width,
				  .height = info.height,
				  .depth  = 1,
				},
			};
		});
		cb.copyBufferToImage(tb.buffer, img.image, vk::ImageLayout::eTransferDstOptimal, copy_regions);

		// image_layout_transition(cb, img.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, sub_res_rng);
		image_layout_transition(cb, img.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

		// End recording
		cb.end();

		// Submit to Transfer Queue
		auto cb_submit_info = vk::CommandBufferSubmitInfo{
			.commandBuffer = cb,
		};
		auto submit_info = vk::SubmitInfo2{
			.commandBufferInfoCount = 1,
			.pCommandBufferInfos    = &cb_submit_info,
		};
		// ctx.gfx_queue.queue.submit2(submit_info, ctx.tfr_in_flight_fence);
		ctx.transfer_queue.queue.submit2(submit_info, ctx.tfr_in_flight_fence);

		// Wait for submission to finish
		auto fence_result = ctx.device.waitForFences(ctx.tfr_in_flight_fence, true, wait_time);
		assert(fence_result == vk::Result::eSuccess and "Failed to wait for transfer fence.");

		std::println("{}Copied GPU texture buffer to GPU image.{}", CLR::CYN, CLR::RESET);
	}

	// Initialize all the per-frame objects
	auto init_frame(const base::vulkan_context &ctx, const shader_binaries &shaders, io::byte_spans ubo_data, const io::texture &tex_data) -> render_context
	{
		std::println("{}Initializing Frame...{}", CLR::CYN, CLR::RESET);
		auto rndr = render_context{};

		create_uniform_descriptor_set(ctx, rndr);
		create_uniform_descriptor_buffer(ctx, rndr);

		create_texture_descriptor_set(ctx, rndr);
		create_texture_descriptor_buffer(ctx, rndr);

		create_pipeline(ctx, rndr, shaders);

		// Create multiple uniform buffers using size of ubo_data
		auto ubo_sizes = ubo_data | std::views::transform([](auto &&span_data) {
			return static_cast<uint32_t>(span_data.size());
		}) | std::ranges::to<std::vector>();

		// Creation of Uniform Buffer only need to know how big the buffer should be.
		// Data is populated later
		create_uniform_buffer(ctx, rndr, ubo_sizes);

		// Creation of texture staging buffer to hold image data
		// actual data populated later, it should be destroyed once vk::Image is populated
		// TODO: destroy the texture_buffer once copy to texture_image is done
		create_texture_buffer(ctx, rndr, tex_data.header_info.size_bytes);

		// Creation of Image which is final destination of texture data
		create_texture_image(ctx, rndr, tex_data.header_info);

		// Creation of Image Sampler for Pixel/Fragment shader
		create_texture_sampler(ctx, rndr);

		// Populate descriptor, uniform and texture buffers
		populate_uniform_descriptor_buffer(ctx, rndr);
		populate_uniform_buffer(ctx, rndr, ubo_data);

		populate_texture_descriptor_buffer(ctx, rndr);
		populate_texture_buffer(ctx, rndr, tex_data.data);
		copy_texture_buffer_to_image(ctx, rndr, tex_data.sub_info);

		return rndr;
	}

	// Clean up all the per-frame objects
	// Order of destruction is important
	void destroy_frame(const base::vulkan_context &ctx, render_context &rndr)
	{
		std::println("{}Destroying Frame...{}", CLR::CYN, CLR::RESET);

		ctx.device.waitIdle();

		// Destroy texture sampler
		ctx.device.destroySampler(rndr.texture_sampler);

		// Destroy texture image
		ctx.device.destroyImageView(rndr.texture_image.view);
		ctx.mem_allocator.destroyImage(rndr.texture_image.image, rndr.texture_image.allocation);
		rndr.texture_image = {};

		// Destroy texture buffer
		ctx.mem_allocator.destroyBuffer(rndr.texture_buffer.buffer, rndr.texture_buffer.allocation);
		rndr.texture_buffer = {};

		// Destroy uniform buffers
		for (auto &&ubo : rndr.uniform_buffers)
		{
			ctx.mem_allocator.destroyBuffer(ubo.buffer, ubo.allocation);
		}
		rndr.uniform_buffers.clear();

		// Destroy descriptor buffer
		ctx.mem_allocator.destroyBuffer(rndr.uniform_descriptor.buffer.buffer, rndr.uniform_descriptor.buffer.allocation);
		ctx.mem_allocator.destroyBuffer(rndr.texture_descriptor.buffer.buffer, rndr.texture_descriptor.buffer.allocation);

		// Destroy pipeline and layout
		ctx.device.destroyPipelineLayout(rndr.layout);
		ctx.device.destroyPipeline(rndr.pipeline);

		// Destroy descriptor set layout
		ctx.device.destroyDescriptorSetLayout(rndr.uniform_descriptor.layout);
		rndr.uniform_descriptor = {};
		ctx.device.destroyDescriptorSetLayout(rndr.texture_descriptor.layout);
		rndr.texture_descriptor = {};
	}

	/**
	 * Functions below get called Every Frame
	 */

	// Prepare command buffer for drawing a frame
	void update_command_buffer(const base::vulkan_context &ctx, const render_context &rndr)
	{
		// Get Sync objects for current frame
		auto sync = ctx.image_signals.at(ctx.current_frame);

		// Wait for Fence to trigger for this frame
		auto fence_result = ctx.device.waitForFences(sync.in_flight_fence,
		                                             true,
		                                             wait_time);
		assert(fence_result == vk::Result::eSuccess and "Failed to wait for graphics fence");

		// Reset Fence
		ctx.device.resetFences(sync.in_flight_fence);

		// Acquire image for current frame
		auto [result, image_index] = ctx.device.acquireNextImageKHR(ctx.swap_chain,
		                                                            wait_time,
		                                                            sync.image_available,
		                                                            VK_NULL_HANDLE);
		assert((result == vk::Result::eSuccess or result == vk::Result::eSuboptimalKHR) // Suboptimal is happens when window is resized.
		       and "Failed to acquire next image");
		assert(image_index == ctx.current_frame and "Image index mismatch"); // Image index should match current frame, if logic is correct.

		// current command buffer
		auto cb = ctx.gfx_command_buffers.at(ctx.current_frame);

		// Begin Command Buffer
		auto cb_begin_info = vk::CommandBufferBeginInfo{};
		auto cb_result     = cb.begin(&cb_begin_info);
		assert(cb_result == vk::Result::eSuccess and "Failed to begin command buffer");

		// current draw image
		auto draw_image = ctx.sc_images.at(ctx.current_frame);

		// Color Range
		auto color_range = vk::ImageSubresourceRange{
			.aspectMask     = vk::ImageAspectFlagBits::eColor,
			.baseMipLevel   = 0,
			.levelCount     = vk::RemainingMipLevels,
			.baseArrayLayer = 0,
			.layerCount     = vk::RemainingArrayLayers,
		};

		// Transition Image Layout
		image_layout_transition(
			cb,
			draw_image,
			image_transition_info{
			  .src_stage_mask    = vk::PipelineStageFlagBits::eColorAttachmentOutput,
			  .dst_stage_mask    = vk::PipelineStageFlagBits::eColorAttachmentOutput,
			  .src_access_mask   = vk::AccessFlags{},
			  .dst_access_mask   = vk::AccessFlagBits::eColorAttachmentWrite,
			  .old_layout        = vk::ImageLayout::eUndefined,
			  .new_layout        = vk::ImageLayout::eColorAttachmentOptimal,
			  .subresource_range = color_range,
			});

		// Color to clear the image with
		auto clear_value = vk::ClearValue{
			.color = rndr.clear_color,
		};

		// Color Attachment
		auto color_attachment = vk::RenderingAttachmentInfo{
			.imageView   = ctx.sc_views.at(ctx.current_frame),
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.resolveMode = vk::ResolveModeFlagBits::eNone,
			.loadOp      = vk::AttachmentLoadOp::eClear,
			.storeOp     = vk::AttachmentStoreOp::eStore,
			.clearValue  = clear_value,
		};

		// Scissors
		auto scissor = vk::Rect2D{
			.offset = { .x = 0, .y = 0 },
			.extent = ctx.sc_extent, // Scissor covers the entire swapchain image
		};
		auto scissors = std::array{ scissor };

		// Begin Rendering
		auto rendering_info = vk::RenderingInfo{
			.renderArea           = scissor,
			.layerCount           = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments    = &color_attachment,
		};
		cb.beginRendering(rendering_info);

		// Viewport
		auto viewport = vk::Viewport{
			.x        = 0.0f,
			.y        = static_cast<float>(ctx.sc_extent.height),        // Flip Y axis, by setting Y-start to height
			.width    = static_cast<float>(ctx.sc_extent.width),         //
			.height   = static_cast<float>(ctx.sc_extent.height) * -1.f, // so it goes from -height to 0, so we can get LHS coordinate system
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		auto viewports = std::array{ viewport };

		// Set Viewport
		cb.setViewport(0, viewports);

		// Set Scissor
		cb.setScissor(0, scissors);

		// Set Pipeline
		cb.bindPipeline(vk::PipelineBindPoint::eGraphics, rndr.pipeline);

		// Descriptor Buffer Bindings data
		auto desc_buff_binding_info = std::array{
			vk::DescriptorBufferBindingInfoEXT{
			  .address = rndr.uniform_descriptor.buffer.address,                // Address of the descriptor buffer
			  .usage   = vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT, // Type of Descriptor Buffer, must match descriptor buffer usage
			},
			vk::DescriptorBufferBindingInfoEXT{
			  .address = rndr.texture_descriptor.buffer.address, // Address of the descriptor buffer
			  .usage   = vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT |
			           vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT, // Type of Descriptor Buffer, must match descriptor buffer usage
			},
		};

		// Bind Descriptor buffers
		cb.bindDescriptorBuffersEXT(desc_buff_binding_info);

		auto desc_buff_set_idx = 0u; // index of uniform_descriptor in desc_buff_binding_info above
		auto desc_buff_offset  = vk::DeviceSize{ 0 };
		// Set location of Projection data for shader, determined by desc_buff_offset
		cb.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics,
		                                 rndr.layout,
		                                 0, // Binding: from descriptor set layout, Set: 0
		                                 1,
		                                 &desc_buff_set_idx,
		                                 &desc_buff_offset);

		// Set location of Transform data for shader, determined by desc_buff_offset
		desc_buff_offset += rndr.uniform_descriptor.layout_size;
		cb.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics,
		                                 rndr.layout,
		                                 1, // Binding: from descriptor set layout, Set: 1
		                                 1,
		                                 &desc_buff_set_idx,
		                                 &desc_buff_offset);

		// Set location of Texture data for shader, determined by desc_buff_offset
		desc_buff_set_idx = 1; // index of texture_descriptor in desc_buff_binding_info above
		desc_buff_offset  = 0;
		cb.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics,
		                                 rndr.layout,
		                                 2, // Binding: from descriptor set layout, Set: 2
		                                 1,
		                                 &desc_buff_set_idx,
		                                 &desc_buff_offset);

		// Draw the triangle, verticies for which is embedded in Vertex Shader
		cb.draw(3, 3, 0, 0);

		// End Rendering
		cb.endRendering();

		// Transition Image Layout
		image_layout_transition(cb,
		                        draw_image,
		                        vk::ImageLayout::eColorAttachmentOptimal,
		                        vk::ImageLayout::ePresentSrcKHR,
		                        color_range);

		// End Command Buffer
		cb.end();
	}

	// Submit and present the frame
	void submit_and_present(base::vulkan_context &ctx)
	{
		auto sync = ctx.image_signals.at(ctx.current_frame);       // Get Sync objects for current frame
		auto cb   = ctx.gfx_command_buffers.at(ctx.current_frame); // Get Command Buffer for current frame

		auto wait_stage = vk::PipelineStageFlags{ vk::PipelineStageFlagBits::eColorAttachmentOutput };

		// Submit Command Buffer
		auto submit_info = vk::SubmitInfo{
			.waitSemaphoreCount   = 1,
			.pWaitSemaphores      = &sync.image_available,
			.pWaitDstStageMask    = &wait_stage,
			.commandBufferCount   = 1,
			.pCommandBuffers      = &cb,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores    = &sync.render_finished,
		};
		ctx.gfx_queue.queue.submit({ submit_info }, sync.in_flight_fence);

		// Present Image
		auto present_info = vk::PresentInfoKHR{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores    = &sync.render_finished,
			.swapchainCount     = 1,
			.pSwapchains        = &ctx.swap_chain,
			.pImageIndices      = &ctx.current_frame,
		};
		auto result = ctx.gfx_queue.queue.presentKHR(present_info);
		assert((result == vk::Result::eSuccess or     // Should normally be success
		        result == vk::Result::eSuboptimalKHR) // happens when window resizes or closes
		       and "Failed to present image");

		// Update current frame to next frame
		ctx.current_frame = (ctx.current_frame + 1) % ctx.max_frame_count;
	}
}

/*
 * Any non-vulkan related code
 * in this instance it's to create a perspective projection matrix
 */
namespace app
{
	// Projection Matrix for Shader
	struct projection
	{
		glm::mat4 data;
	};

	auto make_perspective_projection(int width, int height) -> projection
	{
		auto aspect_ratio = width / static_cast<float>(height);
		auto proj         = glm::perspective(glm::radians(45.0f), aspect_ratio, 0.1f, 10.0f);
		return projection{ .data = proj };
	}

	// Transform matrix for each triangle
	struct transform
	{
		glm::mat4 data;
	};

	auto make_transform_matrix(glm::vec3 position) -> transform
	{
		return transform{
			.data = glm::translate(glm::mat4(1.0f), position),
		};
	}
}

// Main entry point
auto main() -> int
{
	// Window properties
	constexpr auto app_name      = "Vulkan Minimal Example"sv;
	constexpr auto window_width  = 1920;
	constexpr auto window_height = 1080;

	// Call GLFW to create a window
	auto window = glfw::make_window(window_width, window_height, app_name);

	// Initialize Vulkan
	auto vk_ctx = base::init_vulkan(window.get());

	// Load precompiled shaders
	auto shaders = frame::shader_binaries{
		.vertex   = io::read_file("shaders/basic_shader.vs_6_4.spv"),
		.fragment = io::read_file("shaders/basic_shader.ps_6_4.spv"),
	};

	// Load Texture Asset
	auto tex_data = io::load_texture("data/uv_grid.dds");

	// Uniform data for shader
	auto proj = app::make_perspective_projection(window_width, window_height);

	auto transforms = std::array{
		app::make_transform_matrix(glm::vec3(0.0f, 0.5f, 4.0f)),
		app::make_transform_matrix(glm::vec3(-0.5f, -0.5f, 4.0f)),
		app::make_transform_matrix(glm::vec3(0.5f, -0.5f, 4.0f)),
	};

	// Put the uniform data into a span array
	auto ubo_data = std::array{
		io::as_byte_span(proj),
		io::as_byte_span(transforms),
	};

	// Initialize the render frame objects
	auto rndr        = frame::init_frame(vk_ctx, shaders, ubo_data, tex_data);
	rndr.clear_color = std::array{ 0.4f, 0.4f, 0.5f, 1.0f };

	// Loop until the user closes the window
	std::println("{}Starting main loop...{}", CLR::MAG, CLR::RESET);
	while (not glfwWindowShouldClose(window.get()))
	{
		// update command buffer for current frame
		frame::update_command_buffer(vk_ctx, rndr);

		// submit command buffer and present image
		frame::submit_and_present(vk_ctx);

		// Poll for and process events
		glfwPollEvents();
	}

	// Destroy the render pipeline
	frame::destroy_frame(vk_ctx, rndr);

	// Shutdown Vulkan
	base::shutdown_vulkan(vk_ctx);
	return 0;
}
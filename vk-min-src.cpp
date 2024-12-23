// Include files must be before the import statement
#include <cassert>
#include <Vulkan/vulkan.hpp>
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#define VMA_IMPLEMENTATION
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>

import std;

using namespace std::literals;

// Colors for the console
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

	// Bright Colors
	constexpr auto BBLK = "\033[1;30m";
	constexpr auto BRED = "\033[1;31m";
	constexpr auto BGRN = "\033[1;32m";
	constexpr auto BYEL = "\033[1;33m";
	constexpr auto BBLU = "\033[1;34m";
	constexpr auto BMAG = "\033[1;35m";
	constexpr auto BCYN = "\033[1;36m";
	constexpr auto BWHT = "\033[1;37m";

	constexpr auto RESET = "\033[0m";
}

/*
 * Platform IO helpers
 * Basic helpers to read a file in binary mode
 */
namespace io
{
	// Read a file in binary mode
	auto read_file(const std::filesystem::path &filename) -> std::vector<std::byte>
	{
		auto file = std::ifstream(filename, std::ios::in | std::ios::binary);

		assert(file.good() && "failed to open file!");

		auto file_size = std::filesystem::file_size(filename);
		auto buffer    = std::vector<std::byte>(file_size);

		file.read(reinterpret_cast<char *>(buffer.data()), file_size);

		file.close();

		return buffer;
	}
}

/*
 * GLFW helpers
 * Basic helpers to create a window and close it on escape
 */
namespace glfw
{
	// Deleter for GLFWWindow to use with UniquePtr
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
			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			{
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}
		});
	}

	// Create a window with GLFW
	auto make_window(int width, int height, std::string_view title) -> std::unique_ptr<GLFWwindow, destroy_glfw_win>
	{
		std::println("{}Creating window...{}", CLR::BLU, CLR::RESET);

		glfwSetErrorCallback(error_callback); // in case of issues print to console

		glfwInit(); // Initialize GLFW

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No OpenGL, we're using Vulkan
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // Window is not resizable

		auto window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr); // Create the window
		assert(window != nullptr && "Failed to create GLFW window");

		glfw::close_window_on_escape(window); // Close the window on escape

		return std::unique_ptr<GLFWwindow, destroy_glfw_win>(window); // Return the window
	}
}

/*
 * Required for Dynamic Dispatch Loader in Vulkan-HPP
 * Must be called EXACTLY once for Dynamic Dispatch Loader to work
 */
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

/*
 * All Vulkan code will be in vkm namespace
 */
namespace vkm
{
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

		// Created by create_command_pool
		vk::CommandPool gfx_command_pool;
		std::vector<vk::CommandBuffer> gfx_command_buffers;
	};

	auto create_instance(vulkan_context &ctx) -> vkb::Instance
	{
		auto builder = vkb::InstanceBuilder{};

		auto vkb_ib_ret = builder.set_app_name("vk-minimal-example")
		                      .request_validation_layers(use_vulkan_validation_layers)
		                      .set_debug_callback(debug_callback)
		                      .require_api_version(1, 3, 0)
		                      .build();
		assert(vkb_ib_ret.has_value() == true && "Failed to create Vulkan Instance");

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

	void create_surface(vulkan_context &ctx, GLFWwindow *window)
	{
		auto result = static_cast<vk::Result>(
			glfwCreateWindowSurface(ctx.instance,
		                            window,
		                            nullptr,
		                            reinterpret_cast<VkSurfaceKHR *>(&ctx.surface)));
		assert(result == vk::Result::eSuccess && "Failed to create Vulkan Surface");

		std::println("{}Surface created.{}", CLR::GRN, CLR::RESET);
	}

	void pick_gpu_and_queues(vulkan_context &ctx, vkb::Instance &vkb_inst)
	{
		auto features1_3 = vk::PhysicalDeviceVulkan13Features{
			.synchronization2 = true,
			.dynamicRendering = true,
		};

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

		auto features = vk::PhysicalDeviceFeatures{
			.shaderInt64 = true,
		};

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
		                          .add_required_extension(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)
		                          .add_required_extension_features(descriptor_buffer_feature)
		                          .set_surface(ctx.surface)
		                          .select();
		if (not phy_dev_select.has_value())
			std::println("Return value: {}", phy_dev_select.error().message());
		assert(phy_dev_select.has_value() == true && "Failed to select Physical GPU");

		auto phy_dev_ret = phy_dev_select.value();
		auto res         = phy_dev_ret.enable_extension_features_if_present(descriptor_buffer_feature);
		assert(res == true && "Failed to enable extension features on GPU");

		auto device_builder = vkb::DeviceBuilder{ phy_dev_ret };
		auto vkb_device     = device_builder.build().value();

		ctx.chosen_gpu = phy_dev_ret.physical_device;
		ctx.device     = vkb_device.device;

		VULKAN_HPP_DEFAULT_DISPATCHER.init(ctx.device); // get device specific function pointers

		ctx.gfx_queue.queue  = vkb_device.get_queue(vkb::QueueType::graphics).value();
		ctx.gfx_queue.family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

		ctx.transfer_queue.queue  = vkb_device.get_queue(vkb::QueueType::transfer).value();
		ctx.transfer_queue.family = vkb_device.get_queue_index(vkb::QueueType::transfer).value();

		ctx.present_queue.queue  = vkb_device.get_queue(vkb::QueueType::present).value();
		ctx.present_queue.family = vkb_device.get_queue_index(vkb::QueueType::present).value();

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

	void create_swapchain(vulkan_context &ctx)
	{
		auto sfc_prop = ctx.chosen_gpu.getSurfaceCapabilitiesKHR(ctx.surface);
		auto width    = sfc_prop.currentExtent.width;
		auto height   = sfc_prop.currentExtent.height;

		auto sc_builder = vkb::SwapchainBuilder{ ctx.chosen_gpu, ctx.device, ctx.surface };
		auto vkb_sc     = sc_builder
		                  .set_desired_format({
							.format     = VK_FORMAT_B8G8R8A8_UNORM,
							.colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
						  })
		                  .set_desired_extent(width, height)
		                  .build()
		                  .value();

		ctx.swap_chain = vkb_sc.swapchain;
		ctx.sc_extent  = vkb_sc.extent;
		ctx.sc_format  = vk::Format{ vkb_sc.image_format };

		std::ranges::for_each(vkb_sc.get_images().value(), [&](auto &&img) {
			ctx.sc_images.push_back(img);
		});

		std::ranges::for_each(vkb_sc.get_image_views().value(), [&](auto &&img_vw) {
			ctx.sc_views.push_back(img_vw);
		});

		ctx.max_frame_count = static_cast<uint32_t>(ctx.sc_views.size());
		ctx.current_frame   = 0;

		std::println("{}Swapchain created.{}\n"
		             "\tImage size: {} x {}\n"
		             "\tImage count: {}",
		             CLR::GRN, CLR::RESET, width, height, ctx.max_frame_count);
	}

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

		std::println("{}Semaphores and Fences created.{}",
		             CLR::GRN, CLR::RESET);
	}

	void create_command_pool(vulkan_context &ctx)
	{
		// Create Command Pool
		auto command_pool_info = vk::CommandPoolCreateInfo{
			.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = ctx.gfx_queue.family,
		};
		ctx.gfx_command_pool = ctx.device.createCommandPool(command_pool_info);

		std::println("{}Graphics Command Pool created.{}",
		             CLR::GRN, CLR::RESET);

		// Create Command Buffer
		auto command_buffer_alloc_info = vk::CommandBufferAllocateInfo{
			.commandPool        = ctx.gfx_command_pool,
			.level              = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = ctx.max_frame_count,
		};
		ctx.gfx_command_buffers = ctx.device.allocateCommandBuffers(command_buffer_alloc_info);

		std::println("{}Command Buffers allocated.{}",
		             CLR::GRN, CLR::RESET);
	}

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

	void shutdown_vulkan(vulkan_context &ctx)
	{
		std::println("{}Shutting down Vulkan...{}", CLR::BLU, CLR::RESET);

		ctx.device.waitIdle();

		// Destroy Command Pool and Buffers
		ctx.gfx_command_buffers.clear();
		ctx.device.destroyCommandPool(ctx.gfx_command_pool);

		// Destroy Semaphores and Fences
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
	struct shader_binaries
	{
		std::vector<std::byte> vertex;
		std::vector<std::byte> fragment;
	};
}

auto main() -> int
{
	// Window properties
	constexpr auto app_name      = "Vulkan Minimal Example"sv;
	constexpr auto window_width  = 1920;
	constexpr auto window_height = 1080;

	// Call GLFW to create a window
	auto window = glfw::make_window(window_width, window_height, app_name);

	// Initialize Vulkan
	auto vk_ctx = vkm::init_vulkan(window.get());

	// Initialize the render pipeline
	auto shaders = vkm::shader_binaries{
		.vertex   = io::read_file("shaders/basic_shader.vs_6_4.spv"),
		.fragment = io::read_file("shaders/basic_shader.ps_6_4.spv"),
	};

	/* Loop until the user closes the window */
	while (not glfwWindowShouldClose(window.get()))
	{
		/* Render here */

		/* Poll for and process events */
		glfwPollEvents();
	}

	// Shutdown Vulkan
	vkm::shutdown_vulkan(vk_ctx);
	return 0;
}
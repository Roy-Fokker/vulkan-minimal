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

		assert(file.good() && "failed to open file!");

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
	template <class T>
	auto as_byte_span(const T &src) -> byte_span
	{
		return std::span{
			reinterpret_cast<const std::byte *>(&src),
			sizeof(T)
		};
	}

	// Covert a any contiguous range type to a span of bytes
	template <std::ranges::contiguous_range T>
	auto as_byte_span(const T &src) -> byte_span
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
			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
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
		assert(window != nullptr && "Failed to create GLFW window");

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

		// Created by create_command_pool
		vk::CommandPool gfx_command_pool;
		std::vector<vk::CommandBuffer> gfx_command_buffers;
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

	// Create a Vulkan Surface using GLFW
	// GLFW abstracts away the platform specific code for Vulkan Surface creation
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
			.shaderInt64 = true,
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
		assert(phy_dev_select.has_value() == true && "Failed to select Physical GPU");

		// Enable the descriptor buffer extension features
		auto phy_dev_ret = phy_dev_select.value();
		auto res         = phy_dev_ret.enable_extension_features_if_present(descriptor_buffer_feature);
		assert(res == true && "Failed to enable extension features on GPU");

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
		ctx.sc_format  = static_cast<vk::Format>(vkb_sc.image_format);

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

		std::println("{}Semaphores and Fences created.{}",
		             CLR::GRN, CLR::RESET);
	}

	// Create a Command Pool and allocate Command Buffers for Graphics use
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
}

/*
 * Vulkan Objects used every frame
 */
namespace frame
{
	// This holds all the data required to render a frame
	struct render_context
	{
		// Created by create_pipeline
		vk::PipelineLayout layout;
		vk::Pipeline pipeline;

		// Populated by main function
		std::array<float, 4> clear_color;

		// Created by create_descriptor_set
		vk::DescriptorSetLayout descriptor_set_layout;
		vk::DeviceSize descriptor_set_layout_size;
		vk::DeviceSize descriptor_set_layout_offset;
		vma::Allocation descriptor_allocation;
		vk::Buffer descriptor_buffer;
		vk::DeviceSize descriptor_buffer_addr;
		vk::PhysicalDeviceDescriptorBufferPropertiesEXT descriptor_buffer_props;

		// Created by create_uniform_buffer
		vma::Allocation ubo_allocation;
		vma::AllocationInfo ubo_info;
		vk::Buffer ubo_buffer;
		vk::DeviceSize ubo_buffer_addr;
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

		// Descriptor Sets to use with the pipeline
		// Only one descriptor set for now
		auto descriptor_sets = std::array{
			rndr.descriptor_set_layout,
		};

		// Pipeline Layout
		auto pipeline_layout_info = vk::PipelineLayoutCreateInfo{
			.setLayoutCount = static_cast<uint32_t>(descriptor_sets.size()),
			.pSetLayouts    = descriptor_sets.data(),
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
		assert(result_value.result == vk::Result::eSuccess && "Failed to create Graphics Pipeline");

		rndr.pipeline = result_value.value;

		// Destroy the shader modules
		for (auto &&[stg, mod] : shader_list)
		{
			ctx.device.destroyShaderModule(mod);
		}

		std::println("{}Pipeline created.{}",
		             CLR::CYN, CLR::RESET);
	}

	// Structure to make image_layout_transition easier to use
	struct image_transition_info
	{
		vk::PipelineStageFlags src_stage_mask;
		vk::PipelineStageFlags dst_stage_mask;
		vk::AccessFlags src_access_mask;
		vk::AccessFlags dst_access_mask;
		vk::ImageLayout old_layout;
		vk::ImageLayout new_layout;
		vk::ImageSubresourceRange subresource_range;
	};

	// Helper functions to get vk::PipelineStageFlags and vk::AccessFlags from vk::ImageLayout
	auto get_pipeline_stage_flags(vk::ImageLayout image_layout) -> vk::PipelineStageFlags
	{
		switch (image_layout)
		{
			using il = vk::ImageLayout;
			using pf = vk::PipelineStageFlagBits;

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
			return pf::eVertexShader | pf::eFragmentShader;
		case il::ePresentSrcKHR:
			return pf::eBottomOfPipe;
		case il::eGeneral:
			assert(false && "Don't know how to get a meaningful vk::PipelineStageFlags for VK_IMAGE_LAYOUT_GENERAL! Don't use it!");
		default:
			assert(false && "Unknown layout flag");
		}
		return {};
	}

	auto get_access_flags(vk::ImageLayout image_layout) -> vk::AccessFlags
	{
		switch (image_layout)
		{
			using il = vk::ImageLayout;
			using af = vk::AccessFlagBits;

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
			assert(false && "Don't know how to get a meaningful vk::AccessFlags for VK_IMAGE_LAYOUT_GENERAL! Don't use it!");
		default:
			assert(false && "Unknown layout flag");
		}
		return {};
	}

	// Create Descriptor Set
	void create_descriptor_set(const base::vulkan_context &ctx, render_context &rndr)
	{
		constexpr auto binding = 0u;

		// return value such that 'size' is adjusted to match the memory 'alignment' requirements of the device
		auto align_size = [](vk::DeviceSize size, vk::DeviceSize alignment) -> vk::DeviceSize {
			return (size + alignment - 1) & ~(alignment - 1);
		};

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

		rndr.descriptor_set_layout        = ctx.device.createDescriptorSetLayout(desc_set_layout_info);
		rndr.descriptor_set_layout_size   = ctx.device.getDescriptorSetLayoutSizeEXT(rndr.descriptor_set_layout);
		rndr.descriptor_set_layout_offset = ctx.device.getDescriptorSetLayoutBindingOffsetEXT(rndr.descriptor_set_layout, binding);

		// Get descriptor buffer properties
		auto device_props = vk::PhysicalDeviceProperties2{
			.pNext = &rndr.descriptor_buffer_props,
		};
		ctx.chosen_gpu.getProperties2(&device_props);

		// Align the size to the required alignment
		rndr.descriptor_set_layout_size = align_size(rndr.descriptor_set_layout_size, rndr.descriptor_buffer_props.descriptorBufferOffsetAlignment);

		std::println("{}Descriptor Set created.{}", CLR::CYN, CLR::RESET);
	}

	// Create Descriptor Buffer and allocation
	void create_descriptor_buffer(const base::vulkan_context &ctx, render_context &rndr)
	{
		auto buffer_info = vk::BufferCreateInfo{
			.size  = rndr.descriptor_set_layout_size,
			.usage = vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		};

		auto alloc_info = vma::AllocationCreateInfo{
			.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped,
			.usage = vma::MemoryUsage::eAuto,
		};

		std::tie(rndr.descriptor_buffer, rndr.descriptor_allocation) = ctx.mem_allocator.createBuffer(buffer_info, alloc_info);

		auto buff_addr_info = vk::BufferDeviceAddressInfo{
			.buffer = rndr.descriptor_buffer,
		};
		rndr.descriptor_buffer_addr = ctx.device.getBufferAddress(buff_addr_info);

		// Give the buffer a name for debugging
		ctx.device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
		  .objectType   = vk::ObjectType::eBuffer,
		  .objectHandle = (uint64_t)(static_cast<VkBuffer>(rndr.descriptor_buffer)),
		  .pObjectName  = "Descriptor Buffer",
		});

		std::println("{}Descriptor Buffer created.{}", CLR::CYN, CLR::RESET);
	}

	void create_uniform_buffer(const base::vulkan_context &ctx, render_context &rndr, uint32_t size)
	{
		auto buffer_info = vk::BufferCreateInfo{
			.size  = size,
			.usage = vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		};

		auto alloc_info = vma::AllocationCreateInfo{
			.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped,
			.usage = vma::MemoryUsage::eAuto,
		};

		std::tie(rndr.ubo_buffer, rndr.ubo_allocation) = ctx.mem_allocator.createBuffer(buffer_info, alloc_info, &rndr.ubo_info);

		auto buff_addr_info = vk::BufferDeviceAddressInfo{
			.buffer = rndr.ubo_buffer,
		};
		rndr.ubo_buffer_addr = ctx.device.getBufferAddress(buff_addr_info);

		// Give the buffer a name for debugging
		ctx.device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
		  .objectType   = vk::ObjectType::eBuffer,
		  .objectHandle = (uint64_t)(static_cast<VkBuffer>(rndr.ubo_buffer)),
		  .pObjectName  = "Uniform Buffer",
		});

		std::println("{}Uniform Buffer created.{}", CLR::CYN, CLR::RESET);
	}

	// Populate Descriptor Buffer with Uniform Buffer Address and Size
	void populate_descriptor_buffer(const base::vulkan_context &ctx, render_context &rndr)
	{
		std::println("{}Populating Descriptor Buffer.{}", CLR::CYN, CLR::RESET);

		auto buff_addr_info = vk::BufferDeviceAddressInfo{
			.buffer = rndr.ubo_buffer,
		};
		auto ubo_buff_addr = ctx.device.getBufferAddress(buff_addr_info);

		auto address_info = vk::DescriptorAddressInfoEXT{
			.address = ubo_buff_addr,
			.range   = rndr.ubo_info.size,
			.format  = vk::Format::eUndefined,
		};

		auto buff_descriptor_info = vk::DescriptorGetInfoEXT{
			.type = vk::DescriptorType::eUniformBuffer,
			.data = {
			  .pUniformBuffer = &address_info,
			},
		};

		auto descriptor_buffer_ptr = ctx.mem_allocator.mapMemory(rndr.descriptor_allocation);
		ctx.device.getDescriptorEXT(&buff_descriptor_info, rndr.descriptor_buffer_props.uniformBufferDescriptorSize, descriptor_buffer_ptr);
		ctx.mem_allocator.unmapMemory(rndr.descriptor_allocation); // Unmap or flush to keep mapped object alive on CPU side.

		std::println("{}Descriptor Buffer populated.{}", CLR::CYN, CLR::RESET);
	}

	// Populate Uniform Buffer with data, in this example Perspective Projection Matrix
	void populate_uniform_buffer(const base::vulkan_context &ctx, render_context &rndr, std::span<const std::byte> data)
	{
		std::println("{}Populating Uniform Buffer.{}", CLR::CYN, CLR::RESET);

		auto ubo_ptr = ctx.mem_allocator.mapMemory(rndr.ubo_allocation);
		std::memcpy(ubo_ptr, data.data(), data.size());
		ctx.mem_allocator.unmapMemory(rndr.ubo_allocation); // Unmap or flush to keep mapped object alive on CPU side.

		std::println("{}Uniform Buffer populated.{}", CLR::CYN, CLR::RESET);
	}

	// Initialize all the per-frame objects
	auto init_frame(const base::vulkan_context &ctx, const shader_binaries &shaders, uint32_t ubo_size) -> render_context
	{
		std::println("{}Initializing Frame...{}", CLR::CYN, CLR::RESET);
		auto rndr = render_context{};

		create_descriptor_set(ctx, rndr);
		create_pipeline(ctx, rndr, shaders);
		create_descriptor_buffer(ctx, rndr);
		create_uniform_buffer(ctx, rndr, ubo_size);

		populate_descriptor_buffer(ctx, rndr);

		return rndr;
	}

	// Clean up all the per-frame objects
	// Order of destruction is important
	void destroy_frame(const base::vulkan_context &ctx, render_context &rndr)
	{
		std::println("{}Destroying Frame...{}", CLR::CYN, CLR::RESET);

		ctx.device.waitIdle();

		// Destroy uniform buffer and descriptor buffer
		ctx.mem_allocator.destroyBuffer(rndr.ubo_buffer, rndr.ubo_allocation);
		ctx.mem_allocator.destroyBuffer(rndr.descriptor_buffer, rndr.descriptor_allocation);

		// Destroy pipeline and layout
		ctx.device.destroyPipelineLayout(rndr.layout);
		ctx.device.destroyPipeline(rndr.pipeline);

		// Destroy descriptor set layout
		ctx.device.destroyDescriptorSetLayout(rndr.descriptor_set_layout);
	}

	// Functions below get called Every Frame

	// Add Pipeline Barrier to transition image from old_layout to new_layout
	void image_layout_transition(vk::CommandBuffer cb, vk::Image image, const image_transition_info &iti)
	{
		auto image_memory_barrier = vk::ImageMemoryBarrier{
			.srcAccessMask       = iti.src_access_mask,
			.dstAccessMask       = iti.dst_access_mask,
			.oldLayout           = iti.old_layout,
			.newLayout           = iti.new_layout,
			.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
			.dstQueueFamilyIndex = vk::QueueFamilyIgnored,
			.image               = image,
			.subresourceRange    = iti.subresource_range,
		};

		cb.pipelineBarrier(iti.src_stage_mask, iti.dst_stage_mask,
		                   vk::DependencyFlags{},
		                   {}, {},
		                   { image_memory_barrier });
	}

	// overload image_layout_transition with fewer parameters
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

	// Prepare command buffer for drawing a frame
	void update_command_buffer(const base::vulkan_context &ctx, const render_context &rndr)
	{
		// Maximum time to wait for fence
		constexpr auto wait_time = UINT_MAX;

		// Get Sync objects for current frame
		auto sync = ctx.image_signals.at(ctx.current_frame);

		// Wait for Fence to trigger for this frame
		auto fence_result = ctx.device.waitForFences(sync.in_flight_fence,
		                                             true,
		                                             wait_time);
		assert(fence_result == vk::Result::eSuccess && "Failed to wait for fence");

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
		assert(cb_result == vk::Result::eSuccess && "Failed to begin command buffer");

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
		auto desc_buff_binding_info = vk::DescriptorBufferBindingInfoEXT{
			.address = rndr.descriptor_buffer_addr,                           // Address of the descriptor buffer
			.usage   = vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT, // Type of Descriptor Buffer, must match descriptor bugger usage
		};
		auto desc_buff_set_idx = 0u;
		auto desc_buff_offset  = vk::DeviceSize{ 0 };

		// Bind Descriptor buffer
		cb.bindDescriptorBuffersEXT(1, &desc_buff_binding_info);
		cb.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics,
		                                 rndr.layout,
		                                 0,
		                                 1,
		                                 &desc_buff_set_idx,
		                                 &desc_buff_offset);

		// Draw the triangle, verticies for which is embedded in Vertex Shader
		cb.draw(3, 1, 0, 0);

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

	// Initialize the render frame objects
	auto rndr        = frame::init_frame(vk_ctx, shaders, sizeof(app::projection));
	rndr.clear_color = std::array{ 0.5f, 0.4f, 0.5f, 1.0f };

	// Uniform data for shader
	auto proj = app::make_perspective_projection(window_width, window_height);
	frame::populate_uniform_buffer(vk_ctx, rndr, io::as_byte_span(proj));

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
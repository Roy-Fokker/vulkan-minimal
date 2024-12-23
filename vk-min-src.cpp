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
 * All Vulkan code will be is vkm namespace
 */
namespace vkm
{

}

auto main() -> int
{
	// Window properties
	constexpr auto app_name      = "Vulkan Minimal Example"sv;
	constexpr auto window_width  = 1920;
	constexpr auto window_height = 1080;

	// Call GLFW to create a window
	auto window = glfw::make_window(window_width, window_height, app_name);


	/* Loop until the user closes the window */
	while (not glfwWindowShouldClose(window.get()))
	{
		/* Render here */

		/* Poll for and process events */
		glfwPollEvents();
	}
	return 0;
}
// Include files must be before the import statement
#include <Vulkan/vulkan.hpp>
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#define VMA_IMPLEMENTATION
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>

import std;

using namespace std::literals;

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
			glfwDestroyWindow(ptr);
			glfwTerminate();
		}
	};

	// Callback for GLFW errors
	void error_callback(int error, const char *description)
	{
		std::println("Error {}: {}", error, description);
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
	auto make_window(int width, int height, std::string_view title) -> std::expected<std::unique_ptr<GLFWwindow, destroy_glfw_win>, int>
	{
		glfwSetErrorCallback(error_callback); // in case of issues print to console

		glfwInit(); // Initialize GLFW

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No OpenGL, we're using Vulkan
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // Window is not resizable

		auto window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr); // Create the window

		if (window == nullptr) // Check if the window was created
		{
			glfwTerminate();
			std::println("Failed to create window");
			return std::unexpected(-1); // failed to create window
		}

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
	auto wnd_exp = glfw::make_window(window_width, window_height, app_name);
	if (not wnd_exp.has_value())
	{
		return wnd_exp.error();
	}
	auto window = std::move(wnd_exp.value());

	/* Loop until the user closes the window */
	while (not glfwWindowShouldClose(window.get()))
	{
		/* Render here */

		/* Poll for and process events */
		glfwPollEvents();
	}
	return 0;
}
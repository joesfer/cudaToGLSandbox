find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_library(imgui STATIC
	imgui.cpp
	imgui_demo.cpp
	imgui_draw.cpp
	imgui_impl_opengl2.cpp
	imgui_impl_glfw.cpp
	imgui_tables.cpp
	imgui_widgets.cpp
	)

target_default_setup(imgui)
target_default_includes(imgui)

target_link_libraries(imgui PRIVATE
	glfw
	${OPENGL_LIBRARIES}
	)

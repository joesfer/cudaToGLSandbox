add_executable(sandbox
	gpu/info.cu
	gpu/kernel.cu
	app.cu
	main.cpp
	)

target_default_setup(sandbox)
target_default_includes(sandbox)

target_include_directories(sandbox PRIVATE
	${PROJECT_SOURCE_DIR}/src/third-party
	)

target_link_libraries(sandbox
	imgui
	)

# Install
install(TARGETS sandbox
	EXPORT ${PROJECT_NAME}Targets
	RUNTIME DESTINATION bin
	)

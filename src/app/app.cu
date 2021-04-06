#include <cstdio>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl2.h"

#include "defines.h"
#include "gpu/info.h"
#include "gpu/kernel.h"

#include "app.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

Application::~Application()
{
	cleanupCuda();
	cleanupGL();
}

bool Application::init() { return initGL() && initCuda(); }

void Application::mainLoop()
{

	// Main loop
	while (!glfwWindowShouldClose(m_window))
	{
		compute();
		drawFrame();
	}
}

bool Application::initGL()
{
	// Create resources

	if (!glfwInit())
	{
		return false;
	}

	m_window = glfwCreateWindow(1024, 768, "Test", NULL, NULL);
	if (!m_window)
	{
		return false;
	}

	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(1); // Enable vsync

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL2_Init();

	// Cache display dimensions
	// TODO handle window resizing?
	glfwGetFramebufferSize(m_window, &m_displayWidth, &m_displayHeight);

	// Create texture for CUDA and GL to interop
	// https://github.com/lxc-xx/CudaSample/blob/master/NVIDIA_CUDA-5.5_Samples/3_Imaging/simpleCUDA2GL/main.cpp
	glGenTextures(1, &m_cudaTexture);
	glBindTexture(GL_TEXTURE_2D, m_cudaTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_displayWidth, m_displayHeight, 0,
	             GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	return true;
}

bool Application::initCuda()
{
	gpu::info();

	m_cudaDestResource =
	    static_cast<unsigned int *>(ALLOCATE(getTextureDataSize()));

	cudaGraphicsGLRegisterImage(&m_cudaTextureResource, m_cudaTexture,
	                            GL_TEXTURE_2D,
	                            cudaGraphicsMapFlagsWriteDiscard);
	return true;
}

void Application::compute()
{
	{ // run the Cuda kernel
		// calculate grid size
		dim3 block(16, 16, 1);
		dim3 grid(m_displayWidth / block.x, m_displayHeight / block.y, 1);
		// execute CUDA kernel
		gpu::dispatchKernel(grid, block, 0, m_cudaDestResource, m_displayWidth);
	}

	{ // copy to OpenGL
		// We want to copy m_cudaDestResource data to the texture
		// map buffer objects to get CUDA device pointers
		cudaArray *texturePtr;
		cudaGraphicsMapResources(1, &m_cudaTextureResource, 0);
		cudaGraphicsSubResourceGetMappedArray(&texturePtr,
		                                      m_cudaTextureResource, 0, 0);

		const size_t pixelSize = sizeof(GLubyte) * 4;
		const size_t widthBytes = pixelSize * m_displayWidth;
		const size_t srcPitchBytes = pixelSize * m_displayWidth;
		const size_t heightRows = m_displayHeight;
		cudaMemcpy2DToArray(texturePtr,         // dst
		                    0,                  // wOffset
		                    0,                  // hOffset
		                    m_cudaDestResource, // src
		                    srcPitchBytes,      // source picth
		                    widthBytes,         // width
		                    heightRows,         // height,
		                    cudaMemcpyDefault);

		cudaGraphicsUnmapResources(1, &m_cudaTextureResource, 0);
	}

	cudaDeviceSynchronize();
}

void Application::drawFrame()
{
	glfwPollEvents();

	// Start the Dear ImGui frame
	ImGui_ImplOpenGL2_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	{ // Show some stats
		ImGui::Begin("Stats");
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
		            1000.0f / ImGui::GetIO().Framerate,
		            ImGui::GetIO().Framerate);
		ImGui::End();
	}

	// Rendering
	ImGui::Render();
	glViewport(0, 0, m_displayWidth, m_displayHeight);
	glClear(GL_COLOR_BUFFER_BIT);

	{ // draw texture

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glViewport(0, 0, m_displayWidth, m_displayHeight);

		glBindTexture(GL_TEXTURE_2D, m_cudaTexture);
		glEnable(GL_TEXTURE_2D);

		glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0);
		glVertex3f(-1.0, -1.0, 0.5);
		glTexCoord2f(1.0, 0.0);
		glVertex3f(1.0, -1.0, 0.5);
		glTexCoord2f(1.0, 1.0);
		glVertex3f(1.0, 1.0, 0.5);
		glTexCoord2f(0.0, 1.0);
		glVertex3f(-1.0, 1.0, 0.5);
		glEnd();

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();

		glDisable(GL_TEXTURE_2D);
	}

	ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

	glfwMakeContextCurrent(m_window);
	glfwSwapBuffers(m_window);
}

void Application::cleanupCuda()
{
	FREE(m_cudaDestResource);
	cudaDeviceReset();
}

void Application::cleanupGL()
{
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

size_t Application::getTextureDataSize() const
{
	int numTexels = m_displayWidth * m_displayHeight;
	int numValues = numTexels * 4;
	int texDataBytes = sizeof(GLubyte) * numValues;
	return texDataBytes;
}

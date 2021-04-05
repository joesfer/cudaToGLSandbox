#pragma once

#include <GLFW/glfw3.h>

class Application
{
public:
	~Application();

	bool init();

	void mainLoop();

private:
	bool initGL();

	bool initCuda();

	void compute();

	void drawFrame();

	void cleanupCuda();

	void cleanupGL();

	size_t getTextureDataSize() const;

private:
	GLFWwindow *m_window = nullptr;
	int m_displayWidth;
	int m_displayHeight;

	GLuint m_cudaTexture;
	struct cudaGraphicsResource *m_cudaTextureResource;
	unsigned int *m_cudaDestResource;
};


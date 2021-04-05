#include "app.h"

int main(int, char **)
{
	Application app;

	if (!app.init())
	{
		return 1;
	}

	app.mainLoop();

	return 0;
}


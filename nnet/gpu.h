#pragma once

/* Select one of the accelerators */
#define NVIDIA	0
#define INTEL	1
#define WARP	2
#define ACCEL	NVIDIA

class gpu
{
public:
	gpu();
	~gpu();

	static void getAccels();
	static void setAccels();
};


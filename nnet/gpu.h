// Copyright (c) 2018 Luke Lorenzini, https://www.zinisolutions.com/
// This file is licensed under the MIT license.
// See the LICENSE file in the project root for more information.

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


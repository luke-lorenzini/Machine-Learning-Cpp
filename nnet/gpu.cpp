#include "stdafx.h"
#include "gpu.h"

gpu::gpu()
{	
}

gpu::~gpu()
{
}

void gpu::getAccels()
{
	std::vector<concurrency::accelerator> accelerators = concurrency::accelerator::get_all();

	for_each(begin(accelerators), end(accelerators), [=](concurrency::accelerator acc)
	{
		std::wcout << "New accelerator: " << acc.description << std::endl;
		std::wcout << "	is_debug = " << acc.is_debug << std::endl;
		std::wcout << "	is_emulated = " << acc.is_emulated << std::endl;
		std::wcout << "	dedicated_memory = " << acc.dedicated_memory << std::endl;
		std::wcout << "	device_path = " << acc.device_path << std::endl;
		std::wcout << "	has_display = " << acc.has_display << std::endl;
		std::wcout << "	double precision = " << acc.supports_double_precision << std::endl;
		std::wcout << "	version = " << (acc.version >> 16) << '.' << (acc.version & 0xFFFF) << std::endl;
	});

	concurrency::accelerator acc_chosen = accelerators[ACCEL];
	concurrency::accelerator::set_default(acc_chosen.device_path);
}

void gpu::setAccels()
{

}

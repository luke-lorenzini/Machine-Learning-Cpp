// Copyright (c) 2018 Luke Lorenzini, https://www.zinisolutions.com/
// This file is licensed under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include "neuron.h"

template <class type_t>
class relu_neuron :
	public neuron<type_t>
{
public:
	relu_neuron(int input, int output);
	~relu_neuron();

protected:
	void activate() override;
	void activate_der() override;
	void init_rand_real(int size, std::vector<type_t> &vect);
};

template <class type_t>
relu_neuron<type_t>::relu_neuron(int input, int output) :
	neuron(input, output)
{
}

template <class type_t>
relu_neuron<type_t>::~relu_neuron()
{
}

template <class type_t>
void relu_neuron<type_t>::activate()
{
	nnet_math<type_t>::relu(ar_z, ar_y);
}

template <class type_t>
void relu_neuron<type_t>::activate_der()
{
	nnet_math<type_t>::relu_der(ar_z, ar_t_e0);
}

template <class type_t>
void relu_neuron<type_t>::init_rand_real(int size, std::vector<type_t> &vect)
{
#ifdef _USE_FIXED_RAND
	/* Always generate the same random numbers */
	std::mt19937 gen(1);
#else
	/* Always generate different random numbers */
	std::mt19937 gen(rd());
#endif

	std::uniform_real_distribution<type_t> distReal(-1, 1);
	const type_t sigma = sqrt((type_t)2 / size);
	std::normal_distribution<type_t> distNorm(0, sigma);

	for (int i = 0; i < size; i++)
	{
		vect.push_back(distReal(gen));
	}
}

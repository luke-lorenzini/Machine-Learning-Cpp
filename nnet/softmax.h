// Copyright (c) 2018 Luke Lorenzini, https://www.zinisolutions.com/
// This file is licensed under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include "neuron.h"

template <class type_t>
class softmax :
	public neuron<type_t>
{
public:
	softmax(int input, int output);
	~softmax();

protected:
	void activate() override;
	void activate_der() override;
};

template <class type_t>
softmax<type_t>::softmax(int input, int output) :
	neuron(input, output)
{
}

template <class type_t>
softmax<type_t>::~softmax()
{
}

template <class type_t>
void softmax<type_t>::activate()
{
	// Extract the maxium value and convert to type type_t
	auto maxi = std::max_element(std::begin(z), std::end(z));
	type_t mmm = *maxi;

	nnet_math<type_t>::exponent(ar_z, ar_t_y, -mmm);

	type_t sum = concurrency::parallel_reduce(begin(t_y), end(t_y), 0, std::plus<type_t>());

	nnet_math<type_t>::softmax(ar_t_y, sum, ar_y);
}

template <class type_t>
void softmax<type_t>::activate_der()
{
#if 0
	/*
	https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
	https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
	*/
	std::vector<type_t> Jacobian;

	// Jacobian
	for (auto i = 0; i < ar_y.extent[0]; ++i)
	{
		for (auto j = 0; j < ar_y.extent[0]; ++j)
		{
			if (i == j)
			{
				Jacobian.push_back(y[i] * (1 - y[j]));
			}
			else
			{
				Jacobian.push_back(-y[i] * y[j]);
			}
		}
	}

	// Jacobian * dy (derrivative of cost function)
	concurrency::array_view<type_t, RANK> ar_Jacobian(ar_y.extent[0], ar_y.extent[0], Jacobian);

#ifdef _USE_TILES
	nnet_math<type_t>::matrix_mult_tile(ar_Jacobian, ar_y, ar_t_e0);
#else
	nnet_math<type_t>::matrix_mult(ar_Jacobian, ar_y, ar_t_e0);
#endif
#else
	//ar_t_e0 = ar_ones;
	concurrency::copy(ar_ones, ar_t_e0);
#endif
}

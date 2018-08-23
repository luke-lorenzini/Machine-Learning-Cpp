#pragma once
#include "neuron.h"

template <class type_t>
class softmax_neuron :
	public neuron<type_t>
{
public:
	softmax_neuron(int input, int output);
	~softmax_neuron();

protected:
	void activate();
	void activate_der();
};

template <class type_t>
softmax_neuron<type_t>::softmax_neuron(int input, int output) :
	neuron(input, output)
{
}

template <class type_t>
softmax_neuron<type_t>::~softmax_neuron()
{
}

template <class type_t>
void softmax_neuron<type_t>::activate()
{
	auto maxi = std::max_element(std::begin(t_y), std::end(t_y));
	type_t mmm = *maxi;

	nnet_math<type_t>::exponent(ar_z, ar_t_y, -mmm);

	type_t sum = concurrency::parallel_reduce(begin(t_y), end(t_y), 0, std::plus<type_t>());

	nnet_math<type_t>::softmax(ar_t_y, (sum + 0.00001), ar_y);
}

template <class type_t>
void softmax_neuron<type_t>::activate_der()
{
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
				Jacobian.push_back(y[i] * (1 - y[i]));
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
}

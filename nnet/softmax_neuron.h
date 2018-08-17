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
	nnet_math<type_t>::exponent(ar_z, ar_t_y);

	type_t sum = concurrency::parallel_reduce(begin(t_y), end(t_y), 0, std::plus<type_t>());

	nnet_math<type_t>::softmax(ar_t_y, sum, ar_y);
}

template <class type_t>
void softmax_neuron<type_t>::activate_der()
{
	nnet_math<type_t>::matrix_sub(ar_ones, ar_y, ar_t_y);

	nnet_math<type_t>::matrix_prod(ar_y, ar_t_y, ar_t_e0);
}

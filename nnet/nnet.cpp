// Copyright (c) 2018 Luke Lorenzini, https://www.zinisolutions.com/
// This file is licensed under the MIT license.
// See the LICENSE file in the project root for more information.

#include "stdafx.h"
#include "gpu.h"
#include "logistic_neuron.h"
#include "neuron.h"
#include "nnet.h"
#include "nnet_math.h"
#include "softmax_neuron.h"
#include "relu_neuron.h"
#include "tanh_neuron.h"

nnet::nnet(input_parms &parms)
{
	OUTPUT_CLASSES = parms.output_classes;
	IN_SIZE = parms.in_size;
	OUT_SIZE = parms.out_size;
	COLS = parms.cols;
	EPOCHS = parms.epochs;

	x_rows = IN_SIZE;
	x_cols = COLS;
	error_out_rows = OUTPUT_CLASSES;
	error_out_cols = COLS;
	error_out_size = error_out_rows * error_out_cols;
	t_rows = OUTPUT_CLASSES;
	t_cols = COLS;
}

nnet::~nnet()
{
}

void nnet::run_sequential(input_data &data)
{
	auto error_den = 1.0;
	auto total_error = 1.0;

	gpu::getAccels();

	std::vector<type_t> error_out(error_out_size, 0);
	concurrency::array_view<type_t, 2> ar_error_out(error_out_rows, error_out_cols, error_out);

	relu_neuron<type_t> neuron_in(OUT_SIZE, IN_SIZE);
	//relu_neuron<type_t> neuron_int0(OUT_SIZE, OUT_SIZE);
	//relu_neuron<type_t> neuron_int1(OUT_SIZE, OUT_SIZE);
	softmax_neuron<type_t> neuron_out(OUTPUT_CLASSES, OUT_SIZE);

	neurons.push_back(&neuron_in);
	//neurons.push_back(&neuron_int0);
	//neurons.push_back(&neuron_int1);
	neurons.push_back(&neuron_out);

	auto neuron_count = (neurons.size() < 0) ? 0 : signed(neurons.size());

	std::chrono::steady_clock::time_point start_time(std::chrono::steady_clock::now());

	for (auto epochs = 0; epochs < EPOCHS; ++epochs)
	{
		for (auto samples = 0; samples < data.size; ++samples)
		{
			concurrency::array_view<type_t, 2> ar_x(x_rows, x_cols, data.x[samples]);
			concurrency::array_view<type_t, 2> ar_t(t_rows, t_cols, data.t[samples]);

			/* Forward Propogation Step */
			neurons[0]->fwd(ar_x);
			for (auto neur_it = 1; neur_it < neuron_count; ++neur_it)
			{
				neurons[neur_it]->fwd(neurons[neur_it - 1]->get_ar_y());
			}

			/* Error */
			nnet_math<type_t>::matrix_sub(neurons[neuron_count - 1]->get_ar_y(), ar_t, ar_error_out);
			nnet_math<type_t>::scalar_div(ar_error_out, data.size, ar_error_out);
			if ((epochs % ((EPOCHS) / 10) == 0) && (samples == 0))
			{
				auto val = 100 * epochs / (double)EPOCHS;
				//ar_error_out.synchronize();

				for (int e = 0; e < error_out_rows; ++e)
				{
					error_den += abs(error_out[e]);
				}

				total_error = error_den / OUTPUT_CLASSES;

				std::cout << "Progress:" << val << "%	Error:" << total_error << std::endl;

				error_den = 0;
			}

			/* Back Propogation Step */
			neurons[neuron_count - 1]->bkwd(ar_error_out);
			neurons[neuron_count - 1]->set_error();
			for (auto neur_it = neuron_count - 2; neur_it >= 0; --neur_it)
			{
				neurons[neur_it]->bkwd(neurons[neur_it + 1]->get_ar_error());
				neurons[neur_it]->set_error();
			}

			/* Accumulate Error Step */
			for (auto neur_it = neuron_count - 1; neur_it > 0; --neur_it)
			{
				neurons[neur_it]->accm(neurons[neur_it - 1]->get_ar_y());
			}
			neurons[0]->accm(ar_x);
		}

		/* Update Weights */
		for (auto neur_it = neuron_count - 1; neur_it >= 0; --neur_it)
		{
			neurons[neur_it]->updt(data.size);
		}
	}
	std::chrono::steady_clock::time_point end_time(std::chrono::steady_clock::now());

	std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count() << std::endl;

	/* Verify */
	const auto CHECK0 = 0;
	const auto CHECK1 = 50;
	const auto CHECK2 = 149;

	/* Check 0 */
	concurrency::array_view<type_t, 2> x_check0(x_rows, x_cols, data.x[CHECK0]);
	concurrency::array_view<type_t, 2> t_check0(t_rows, t_cols, data.t[CHECK0]);
	std::vector<type_t> error0(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check0(error_out_rows, error_out_cols, error0);
	neurons[0]->check(x_check0);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check0 is all 0's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check0, error_check0);

	/* Check 1*/
	concurrency::array_view<type_t, 2> x_check1(x_rows, x_cols, data.x[CHECK1]);
	concurrency::array_view<type_t, 2> t_check1(t_rows, t_cols, data.t[CHECK1]);
	std::vector<type_t> error1(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check1(error_out_rows, error_out_cols, error1);
	neurons[0]->check(x_check1);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check0 is all 1's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check1, error_check1);

	/* Check 2*/
	concurrency::array_view<type_t, 2> x_check2(x_rows, x_cols, data.x[CHECK2]);
	concurrency::array_view<type_t, 2> t_check2(t_rows, t_cols, data.t[CHECK2]);
	std::vector<type_t> error2(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check2(error_out_rows, error_out_cols, error2);
	neurons[0]->check(x_check2);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check2 is all 0's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check2, error_check2);
}

void nnet::run_parallel(input_data &data)
{
	auto error_den = 1.0;
	auto total_error = 1.0;

	gpu::getAccels();

	std::vector<type_t> error_out(error_out_size, 0);
	concurrency::array_view<type_t, 2> ar_error_out(error_out_rows, error_out_cols, error_out);

	relu_neuron<type_t> neuron_in(OUT_SIZE, IN_SIZE);
	//relu_neuron<type_t> neuron_int0(OUT_SIZE, OUT_SIZE);
	//relu_neuron<type_t> neuron_int1(OUT_SIZE, OUT_SIZE);
	softmax_neuron<type_t> neuron_out(OUTPUT_CLASSES, OUT_SIZE);

	neurons.push_back(&neuron_in);
	//neurons.push_back(&neuron_int0);
	//neurons.push_back(&neuron_int1);
	neurons.push_back(&neuron_out);

	auto neuron_count = (neurons.size() < 0) ? 0 : signed(neurons.size());

	std::chrono::steady_clock::time_point start_time(std::chrono::steady_clock::now());

	for (auto epochs = 0; epochs < EPOCHS; ++epochs)
	{
		typedef std::pair<std::vector<type_t>, std::vector<type_t>> DataPair;
		concurrency::concurrent_queue<DataPair> mydata;

		for (auto i = 0; i < data.size; ++i)
		{
			mydata.push(DataPair(data.x[i], data.t[i]));
		}

		std::vector<concurrency::accelerator> accels = concurrency::accelerator::get_all();
		concurrency::parallel_for(0, int(accels.size()), [=, &mydata, &total_error, &error_den](const unsigned i)
		{
			std::vector<type_t> error_out(error_out_size, 0);
			concurrency::array_view<type_t, 2> ar_error_out(error_out_rows, error_out_cols, error_out);
			//auto taskCount = 0;

			DataPair dp;
			while (mydata.try_pop(dp))
			{
				concurrency::array_view<type_t, 2> test_x(x_rows, x_cols, dp.first);
				concurrency::array_view<type_t, 2> test_t(t_rows, t_cols, dp.second);

				/* Forward Propogation Step */
				neurons[0]->fwd(test_x);
				for (auto neur_it = 1; neur_it < neuron_count; ++neur_it)
				{
					neurons[neur_it]->fwd(neurons[neur_it - 1]->get_ar_y());
				}

				/* Error */
				nnet_math<type_t>::matrix_sub(neurons[neuron_count - 1]->get_ar_y(), test_t, ar_error_out);
				nnet_math<type_t>::scalar_div(ar_error_out, data.size, ar_error_out);
				/*if ((epochs % ((EPOCHS) / 10) == 0) && (i == 0))
				{
					auto val = 100 * epochs / (double)EPOCHS;
					ar_error_out.synchronize();

					for (int e = 0; e < error_out_rows; ++e)
					{
						error_den += abs(error_out[e]);
					}

					total_error = error_den / OUTPUT_CLASSES;

					std::cout << "Progress:" << val << "%	Error:" << total_error << std::endl;

					error_den = 0;
				}*/

				/* Back Propogation Step */
				neurons[neuron_count - 1]->bkwd(ar_error_out);
				neurons[neuron_count - 1]->set_error();
				for (auto neur_it = neuron_count - 2; neur_it >= 0; --neur_it)
				{
					neurons[neur_it]->bkwd(neurons[neur_it + 1]->get_ar_error());
					neurons[neur_it]->set_error();
				}

				/* Accumulate Error Step */
				for (auto neur_it = neuron_count - 1; neur_it > 0; --neur_it)
				{
					neurons[neur_it]->accm(neurons[neur_it - 1]->get_ar_y());
				}
				neurons[0]->accm(test_x);

				//taskCount++;
			}

			accels[i].default_view.wait();

			//std::wcout << " Finished " << taskCount << " tasks on " << i << std::endl;
		});

		/* Update Weights */
		for (auto neur_it = neuron_count - 1; neur_it >= 0; --neur_it)
		{
			neurons[neur_it]->updt(data.size);
		}
	}
	std::chrono::steady_clock::time_point end_time(std::chrono::steady_clock::now());

	std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count() << std::endl;

	/* Verify */
	const auto CHECK0 = 0;
	const auto CHECK1 = 50;
	const auto CHECK2 = 149;

	/* Check 0 */
	concurrency::array_view<type_t, 2> x_check0(x_rows, x_cols, data.x[CHECK0]);
	concurrency::array_view<type_t, 2> t_check0(t_rows, t_cols, data.t[CHECK0]);
	std::vector<type_t> error0(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check0(error_out_rows, error_out_cols, error0);
	neurons[0]->check(x_check0);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check0 is all 0's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check0, error_check0);

	/* Check 1*/
	concurrency::array_view<type_t, 2> x_check1(x_rows, x_cols, data.x[CHECK1]);
	concurrency::array_view<type_t, 2> t_check1(t_rows, t_cols, data.t[CHECK1]);
	std::vector<type_t> error1(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check1(error_out_rows, error_out_cols, error1);
	neurons[0]->check(x_check1);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check0 is all 1's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check1, error_check1);

	/* Check 2*/
	concurrency::array_view<type_t, 2> x_check2(x_rows, x_cols, data.x[CHECK2]);
	concurrency::array_view<type_t, 2> t_check2(t_rows, t_cols, data.t[CHECK2]);
	std::vector<type_t> error2(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check2(error_out_rows, error_out_cols, error2);
	neurons[0]->check(x_check2);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check2 is all 0's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check2, error_check2);
}

void nnet::verify(input_data& data)
{
	/* Verify */
	const auto CHECK0 = 0;
	const auto CHECK1 = 50;
	const auto CHECK2 = 149;

	/* Check 0 */
	concurrency::array_view<type_t, 2> x_check0(x_rows, x_cols, data.x[CHECK0]);
	concurrency::array_view<type_t, 2> t_check0(t_rows, t_cols, data.t[CHECK0]);
	std::vector<type_t> error0(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check0(error_out_rows, error_out_cols, error0);
	neurons[0]->check(x_check0);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check0 is all 0's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check0, error_check0);

	/* Check 1*/
	concurrency::array_view<type_t, 2> x_check1(x_rows, x_cols, data.x[CHECK1]);
	concurrency::array_view<type_t, 2> t_check1(t_rows, t_cols, data.t[CHECK1]);
	std::vector<type_t> error1(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check1(error_out_rows, error_out_cols, error1);
	neurons[0]->check(x_check1);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check0 is all 1's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check1, error_check1);

	/* Check 2*/
	concurrency::array_view<type_t, 2> x_check2(x_rows, x_cols, data.x[CHECK2]);
	concurrency::array_view<type_t, 2> t_check2(t_rows, t_cols, data.t[CHECK2]);
	std::vector<type_t> error2(error_out_size, 0);
	concurrency::array_view<type_t, 2> error_check2(error_out_rows, error_out_cols, error2);
	neurons[0]->check(x_check2);
	neurons[1]->check(neurons[0]->get_ar_y());
	/* If error_check2 is all 0's, training was correct for sample */
	nnet_math<type_t>::matrix_sub(neurons[1]->get_ar_y(), t_check2, error_check2);
}

#include "stdafx.h"
#include "gpu.h"
#include "logistic_neuron.h"
#include "neuron.h"
#include "nnet.h"
#include "nnet_math.h"
#include "softmax_neuron.h"
#include "relu_neuron.h"
#include "tanh_neuron.h"

//#define _USE_PARALLEL

nnet::nnet(input_parms& parms)
{
	DATA_ROWS = parms.data_rows;
	DATA_COLS = parms.data_cols;
	OUTPUT_CLASSES = parms.output_classes;
	LAYER_MULT = parms.layer_mult;
	IN_SIZE = parms.in_size;
	OUT_SIZE = parms.out_size;
	COLS = parms.cols;
	EPOCHS = parms.epochs;
}

nnet::~nnet()
{
}

void nnet::run(input_data& data)
{
	auto error_den = 1.0;
	auto total_error = 1.0;

#ifndef _USE_PARALLEL
	gpu::getAccels();
#endif

	const auto x_rows = IN_SIZE;
	const auto x_cols = COLS;

	const auto error_out_rows = OUTPUT_CLASSES;
	const auto error_out_cols = COLS;
	const auto error_out_size = error_out_rows * error_out_cols;
	std::vector<type_t> error_out(error_out_size, 0);
	concurrency::array_view<type_t, 2> ar_error_out(error_out_rows, error_out_cols, error_out);

	const auto t_rows = OUTPUT_CLASSES;
	const auto t_cols = COLS;

	logistic_neuron<type_t> neuron_in(OUT_SIZE, IN_SIZE);
	//logistic_neuron<type_t> neuron_int0(OUT_SIZE, OUT_SIZE);
	logistic_neuron<type_t> neuron_out(OUTPUT_CLASSES, OUT_SIZE);

	std::vector<neuron<type_t>*> neurons;
	neurons.push_back(&neuron_in);
	//neurons.push_back(&neuron_int0);
	neurons.push_back(&neuron_out);

	auto neuron_count = (neurons.size() < 0) ? 0 : signed(neurons.size());

	std::chrono::steady_clock::time_point start_time(std::chrono::steady_clock::now());

	for (auto epochs = 0; epochs < EPOCHS; ++epochs)
	{
#ifdef _USE_PARALLEL
		typedef std::pair<std::vector<type_t>, std::vector<type_t>> DataPair;
		concurrency::concurrent_queue<DataPair> mydata;

		for (auto i = 0; i < data.size; ++i)
		{
			mydata.push(DataPair(data.x[i], data.t[i]));
		}

		std::vector<concurrency::accelerator> accels = concurrency::accelerator::get_all();
		concurrency::parallel_for(0, int(accels.size()), [=, &mydata, &total_error, &ar_error_out, &error_den](const unsigned i)
		{
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
#else
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
			if ((epochs % ((EPOCHS) / 10) == 0) && (samples == 0))
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
#endif
		/* Update Weights */
		for (auto neur_it = neuron_count - 1; neur_it >= 0; --neur_it)
		{
			neurons[neur_it]->updt(data.size);
		}
	}
	std::chrono::steady_clock::time_point end_time(std::chrono::steady_clock::now());

	std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count() << std::endl;
}

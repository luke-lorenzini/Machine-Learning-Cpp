#include "stdafx.h"
#include "file.h"
#include "gpu.h"
#include "logistic_neuron.h"
#include "neuron.h"
#include "nnet_math.h"
#include "softmax_neuron.h"
#include "relu_neuron.h"
#include "tanh_neuron.h"

#define _USE_PARALLEL

typedef double type_t;

#define MNIST	0
#define IRIS	4
#define CLASS_DATA	IRIS

/* Number of times to loop over the samples */
const static auto EPOCHS = 10;

#if CLASS_DATA
const static auto DATA_ROWS = 4;
const static auto DATA_COLS = 1;
const static auto OUTPUT_CLASSES = 3;

const auto FILENAME = "..\\test_data\\iris\\irisMod.csv";
#else
const static auto DATA_ROWS = 28;
const static auto DATA_COLS = 28;
const static auto OUTPUT_CLASSES = 10;

const auto FILENAME = "..\\test_data\\MNIST\\mnist_test.csv";
#endif

const static auto LAYER_MULT = 2;
const static auto IN_SIZE = DATA_ROWS * DATA_COLS;
const static auto OUT_SIZE = IN_SIZE * LAYER_MULT;
const static auto COLS = 1;

double error_den = 1;

int main()
{
	/* The number of samples in the dataset */
	static auto sample_size = 10;

	std::vector<std::vector<type_t>> data = file<type_t>::parseCSV2(FILENAME);
	sample_size = data.size();

	std::random_device rd;
#ifdef _USE_FIXED_RAND
	/* Always generate different random numbers */
	std::mt19937 gen(1);
#else
	/* Always generate the same random numbers */
	std::mt19937 gen(rd());
#endif
	std::uniform_real_distribution<type_t> distReal(0, 1);
	std::uniform_int_distribution<> distInt(0, 255);
	std::uniform_int_distribution<> distBin(0, 1);

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
	const auto t_size = t_rows * t_cols;
	std::vector<type_t> t;
	for (int i = 0; i < t_size; ++i)
	{
		t.push_back(distBin(gen));
	}

	const auto one_hot_size = OUTPUT_CLASSES;
	std::vector<std::vector<type_t>> one_hot;

	for (int i = 0; i < OUTPUT_CLASSES; ++i)
	{
		std::vector<type_t> one_hot_row;

		for (int j = 0; j < OUTPUT_CLASSES; ++j)
		{
			if (i == j)
			{
				one_hot_row.push_back(1);
			}
			else
			{
				one_hot_row.push_back(0);
			}
		}
		
		one_hot.push_back(one_hot_row);
	}

	logistic_neuron<type_t> neuron_in(OUT_SIZE, IN_SIZE);
	//logistic_neuron<type_t> neuron_int0(OUT_SIZE, OUT_SIZE);
	logistic_neuron<type_t> neuron_out(OUTPUT_CLASSES, OUT_SIZE);

	std::vector<neuron<type_t>*> neurons;
	neurons.push_back(&neuron_in);
	//neurons.push_back(&neuron_int0);
	neurons.push_back(&neuron_out);

	int neuron_count = (neurons.size() < 0) ? 0 : neurons.size();

	auto total_error = 1.0;

	std::chrono::steady_clock::time_point start_time(std::chrono::steady_clock::now());

	for (auto epochs = 0; epochs < EPOCHS; ++epochs)
	{
#ifdef _USE_PARALLEL
		typedef std::pair<std::vector<type_t>, std::vector<type_t>> DataPair;
		concurrency::concurrent_queue<DataPair> mydata;

		for (auto i = 0; i < sample_size; ++i)
		{
#if CLASS_DATA
			std::vector<type_t>::const_iterator first = data[i].begin();
			std::vector<type_t>::const_iterator last = data[i].end() - 1;
#else
			std::vector<type_t>::const_iterator first = data[i].begin() + 1;
			std::vector<type_t>::const_iterator last = data[i].end();
#endif
			std::vector<type_t> x(first, last);

			mydata.push(DataPair(x, one_hot[data[i][CLASS_DATA]]));
		}

		std::vector<concurrency::accelerator> accels = concurrency::accelerator::get_all();
		concurrency::parallel_for(0, int(accels.size()), [=, &mydata, &total_error](const unsigned i)
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
				//nnet_math<type_t>::matrix_sub(neurons[neuron_count - 1]->get_ar_y(), ar_t, ar_error_out);
				if ((epochs % ((EPOCHS) / 10) == 0) && (i == 0))
				{
					auto val = 100 * epochs / (double)EPOCHS;
					ar_error_out.synchronize();

					for (int e = 0; e < error_out_rows; ++e)
					{
						error_den += abs(error_out[e]);
					}

					total_error = error_den / OUTPUT_CLASSES;

					//std::cout << "Progress:" << val << "%	Error:" << total_error << std::endl;

					error_den = 0;
				}

				/* Back Propogation Step */
				//neurons[neuron_count - 1]->bkwd(ar_error_out);
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
		for (auto samples = 0; samples < sample_size; ++samples)
		{
#if CLASS_DATA
			std::vector<type_t>::const_iterator first = data[samples].begin();
			std::vector<type_t>::const_iterator last = data[samples].end() - 1;
#else
			std::vector<type_t>::const_iterator first = data[samples].begin() + 1;
			std::vector<type_t>::const_iterator last = data[samples].end();
#endif
			std::vector<type_t> x(first, last);
#ifdef _ARRAYS			
			concurrency::array_view<type_t, 2> ar_x(x_rows, x_cols, x);

			/* Forward Propogation Step */
			neurons[0]->fwd(x);
			for (auto neur_it = 1; neur_it < neuron_count; ++neur_it)
			{
				neurons[neur_it]->fwd(neurons[neur_it - 1]->get_y());
			}

			t = one_hot[data[samples][CLASS_DATA]];
			concurrency::array_view<type_t, 2> ar_t(t_rows, t_cols, t);

			/* Error */
			nnet_math<type_t>::matrix_sub(neuron_out.get_y(), t, error_out, error_out_rows, error_out_cols);
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
			neurons[neuron_count - 1]->bkwd(error_out);
			neurons[neuron_count - 1]->set_error();
			for (auto neur_it = neuron_count - 2; neur_it >= 0; --neur_it)
			{
				neurons[neur_it]->bkwd(neurons[neur_it + 1]->get_error());
				neurons[neur_it]->set_error();
			}

			/* Accumulate Error Step */
			for (auto neur_it = neuron_count - 1; neur_it > 0; --neur_it)
			{
				neurons[neur_it]->accm(neurons[neur_it - 1]->get_y());
			}
			neurons[0]->accm(x);

#else
			concurrency::array_view<type_t, 2> ar_x(x_rows, x_cols, x);

			/* Forward Propogation Step */
			neurons[0]->fwd(ar_x);
			for (auto neur_it = 1; neur_it < neuron_count; ++neur_it)
			{				
				neurons[neur_it]->fwd(neurons[neur_it - 1]->get_ar_y());
			}

			t = one_hot[data[samples][CLASS_DATA]];
			concurrency::array_view<type_t, 2> ar_t(t_rows, t_cols, t);

			/* Error */
			nnet_math<type_t>::matrix_sub(neuron_out.get_ar_y(), ar_t, ar_error_out);
			if ((epochs % ((EPOCHS) / 10) == 0) && (samples == 0))
			{
				auto val = 100 * epochs / (double)EPOCHS;
				ar_error_out.synchronize();

				for (int e = 0; e < error_out_rows; ++e)
				{
					error_den += abs(error_out[e]);
				}

				total_error = error_den / OUTPUT_CLASSES;

				std::cout << "Progress:" << val << "%	Error:" << total_error  << std::endl;

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
				neurons[neur_it]->accm(neurons[neur_it-1]->get_ar_y());
			}
			neurons[0]->accm(ar_x);
#endif
		}
#endif
		/* Update Weights */
		for (auto neur_it = neuron_count - 1; neur_it >= 0; --neur_it)
		{
			neurons[neur_it]->updt(sample_size);
		}
	}
	std::chrono::steady_clock::time_point end_time(std::chrono::steady_clock::now());

	std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

	getchar();
}

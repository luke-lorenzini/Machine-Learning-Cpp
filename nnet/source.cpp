#include "stdafx.h"
#include "file.h"
#include "gpu.h"
#include "neuron.h"

#define _USEREALDATA

/* Number of times to loop over the samples */
const static auto EPOCHS = 100;

/* The number of samples in the dataset */
const static auto SAMPLES = 3;

const static auto DATA_ROWS = 28;
const static auto DATA_COLS = 28;
const static auto LAYER_MULT = 2;
const static auto IN_SIZE = DATA_ROWS * DATA_COLS;
const static auto OUT_SIZE = IN_SIZE * LAYER_MULT;
const static auto OUTPUT_CLASSES = 10;
const static auto COLS = 1;

double error_den = 1;

#ifdef _USEREALDATA
const auto FILENAME = "C:\\Users\\Luke\\Documents\\OneDrive\\Documents\\Visual Studio\\Python\\Machine-Learning\\Test-Data\\MNIST\\mnist_test.csv";
#endif

int main()
{
#ifdef _USEREALDATA
	std::vector<std::vector<type_t>> data = file::parseCSV2(FILENAME);
#endif
	std::random_device rd;
#ifdef _USE_FIXED_RAND
	/* Always generate different random numbers */
	std::mt19937 gen(1);
#else
	/* Always generate the same random numbers */
	std::mt19937 gen(rd());
#endif
	std::uniform_int_distribution<> distInt(0, 255);
	std::uniform_int_distribution<> distBin(0, 1);

	gpu::getAccels();

	const auto x_rows = IN_SIZE;
	const auto x_cols = COLS;
	const auto x_size = x_rows * x_cols;
	std::vector<type_t> x;
	for (int i = 1; i < x_size + 1; ++i)
	{
		x.push_back(distInt(gen));
	}

	const auto error_out_rows = OUTPUT_CLASSES;
	const auto error_out_cols = COLS;
	const auto error_out_size = error_out_rows * error_out_cols;
	std::vector<type_t> error_out(error_out_size, 0);

	const auto t_rows = OUTPUT_CLASSES;
	const auto t_cols = COLS;
	const auto t_size = t_rows * t_cols;
	std::vector<type_t> t;
	for (int i = 0; i < t_size; i++)
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

	const auto NEURON_COUNT = 3;

	neuron neuron_in(OUT_SIZE, IN_SIZE);
	neuron neuron_int0(OUT_SIZE, OUT_SIZE);
	/*neuron neuron_int1(OUT_SIZE, OUT_SIZE);
	neuron neuron_int2(OUT_SIZE, OUT_SIZE);
	neuron neuron_int3(OUT_SIZE, OUT_SIZE);
	neuron neuron_int4(OUT_SIZE, OUT_SIZE);
	neuron neuron_int5(OUT_SIZE, OUT_SIZE);
	neuron neuron_int6(OUT_SIZE, OUT_SIZE);
	neuron neuron_int7(OUT_SIZE, OUT_SIZE);*/
	neuron neuron_out(OUTPUT_CLASSES, OUT_SIZE);

	std::vector<neuron*> neurons;
	neurons.push_back(&neuron_in);
	neurons.push_back(&neuron_int0);
	/*neurons.push_back(&neuron_int1);
	neurons.push_back(&neuron_int2);
	neurons.push_back(&neuron_int3);
	neurons.push_back(&neuron_int4);
	neurons.push_back(&neuron_int5);
	neurons.push_back(&neuron_int6);
	neurons.push_back(&neuron_int7);*/
	neurons.push_back(&neuron_out);

	auto total_error = 1.0;

	std::chrono::steady_clock::time_point start_time(std::chrono::steady_clock::now());

	for (auto epochs = 0; epochs < EPOCHS; ++epochs)
	{
		for (auto samples = 0; samples < SAMPLES; ++samples)
		{
#ifdef _USEREALDATA
			std::vector<type_t>::const_iterator first = data[samples].begin() + 1;
			std::vector<type_t>::const_iterator last = data[samples].end();
			std::vector<type_t> x(first, last);
#endif

			neurons[0]->fwd(x);
			for (auto neur_it = 1; neur_it < NEURON_COUNT; ++neur_it)
			{				
				neurons[neur_it]->fwd(neurons[neur_it - 1]->get_y());
			}

#ifdef _USEREALDATA
			t = one_hot[data[samples][0]];
#else
			t = one_hot[7];
#endif
			neuron::matrix_sub(neuron_out.get_y(), t, error_out, error_out_rows, error_out_cols);

			neurons[NEURON_COUNT - 1]->bkwd(error_out);
			for (auto neur_it = NEURON_COUNT - 2; neur_it >= 0; --neur_it)
			{
				neurons[neur_it]->bkwd(neurons[neur_it + 1]->get_error());
				neurons[neur_it]->set_error();
			}

			for (auto neur_it = NEURON_COUNT - 1; neur_it > 0; --neur_it)
			{
				neurons[neur_it]->accm(neurons[neur_it-1]->get_y());
			}
			neurons[0]->accm(x);

			if (samples % ((EPOCHS * SAMPLES) / 10) == 0)
			{
				auto val = 100 * epochs / (double)EPOCHS;
				std::cout << val << "%" << std::endl;

				for (int e = 0; e < error_out_rows; ++e)
				{
					error_den += abs(error_out[e]);
				}

				total_error = error_den / OUTPUT_CLASSES;

				std::cout << total_error << std::endl;

				error_den = 0;
			}
		}

		for (auto neur_it = NEURON_COUNT - 1; neur_it >= 0; --neur_it)
		{
			neurons[neur_it]->updt();
		}
	}

	std::chrono::steady_clock::time_point end_time(std::chrono::steady_clock::now());

	std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

	getchar();
}

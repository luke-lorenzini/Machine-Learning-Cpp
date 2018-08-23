#include "stdafx.h"
#include "file.h"
#include "nnet.h"

#define MNIST	0
#define IRIS	4
#define CLASS_DATA	IRIS

int main()
{
	/* Unique parameters for different data */
#if CLASS_DATA
	const static auto DATA_ROWS = 4;
	const static auto DATA_COLS = 1;
	const static auto OUTPUT_CLASSES = 3;
	const static auto FILENAME = "..\\test_data\\iris\\irisMod.csv";
#else
	const static auto DATA_ROWS = 28;
	const static auto DATA_COLS = 28;
	const static auto OUTPUT_CLASSES = 10;
	const static auto FILENAME = "..\\test_data\\MNIST\\mnist_test.csv";
#endif
	/* Common parameters */
	const static auto LAYER_MULT = 2;
	const static auto IN_SIZE = DATA_ROWS * DATA_COLS;
	const static auto OUT_SIZE = IN_SIZE * LAYER_MULT;
	const static auto COLS = 1;

	/* Hyper Parameter - How many times to run model */
	const static auto EPOCHS = 10;

	/* Vector of data */
	std::vector<std::vector<type_t>> data = file<type_t>::parseCSV(FILENAME);

	/* Ensure size > 0 */
	static auto sample_size = (data.size() < 0) ? 0 : signed(data.size());

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

	std::vector<std::vector<type_t>> x;
	std::vector<std::vector<type_t>> t;

	for (auto samples = 0; samples < sample_size; ++samples)
	{
#if CLASS_DATA
		std::vector<type_t>::const_iterator first = data[samples].begin();
		std::vector<type_t>::const_iterator last = data[samples].end() - 1;
		std::vector<type_t>::const_iterator soln = data[samples].end();
#else
		std::vector<type_t>::const_iterator first = data[samples].begin() + 1;
		std::vector<type_t>::const_iterator last = data[samples].end();
		std::vector<type_t>::const_iterator soln = data[samples].begin();
#endif

		std::vector<type_t> temp_x(first, last);

		x.push_back(temp_x);
		t.push_back(one_hot[data[samples][CLASS_DATA]]);
	}
	
	/* Input Parameters */
	nnet::input_parms myParms;
	myParms.output_classes = OUTPUT_CLASSES;
	myParms.in_size = IN_SIZE;
	myParms.out_size = OUT_SIZE;
	myParms.cols = COLS;
	myParms.epochs = EPOCHS;

	/* Input Data */
	nnet::input_data myData2;
	myData2.x = x;
	myData2.t = t;
	myData2.size = sample_size;

	/* Create the nnet object */
	nnet id(myParms);

	/* Run the thing */
	id.run(myData2);

	/* Verify the thing */
	//id.verify(myData2);

	/* Wait for some input */
	getchar();
}

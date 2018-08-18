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
	std::vector<std::vector<type_t>> data = file<type_t>::parseCSV2(FILENAME);

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
	
	/* Input Parameters */
	nnet::input_parms myData;
	myData.data_rows = DATA_ROWS;
	myData.data_cols = DATA_COLS;
	myData.output_classes = OUTPUT_CLASSES;
	myData.filename = FILENAME;
	myData.layer_mult = LAYER_MULT;
	myData.in_size = IN_SIZE;
	myData.out_size = OUT_SIZE;
	myData.cols = COLS;

	/* Create the nnet object */
	nnet id(myData);

	/* Run the thing */
	id.run(EPOCHS, data, one_hot);

	/* Wait for some input */
	getchar();
}

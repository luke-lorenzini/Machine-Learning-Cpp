#pragma once
#include "neuron.h"

typedef double type_t;

class nnet
{
public:
	struct input_parms {
		int data_rows;
		int data_cols;
		int output_classes;
		int layer_mult;
		int in_size;
		int out_size;
		int cols;
		int epochs;
	};

	struct input_data {
		std::vector<std::vector<type_t>> x;
		std::vector<std::vector<type_t>> t;
		int size;
	};

	nnet(input_parms& parms);
	~nnet();

	void run(input_data& data);
	void verify(input_data& data);

private:
	int DATA_ROWS;
	int DATA_COLS;
	int OUTPUT_CLASSES;
	int LAYER_MULT;
	int IN_SIZE;
	int OUT_SIZE;
	int COLS;
	int EPOCHS;

	int x_rows = IN_SIZE;
	int x_cols = COLS;
	int error_out_rows = OUTPUT_CLASSES;
	int error_out_cols = COLS;
	int error_out_size = error_out_rows * error_out_cols;
	int t_rows = OUTPUT_CLASSES;
	int t_cols = COLS;

	std::vector<neuron<type_t>*> neurons;
};


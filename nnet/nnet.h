#pragma once

typedef double type_t;

class nnet
{
public:
	struct input_parms {
		int data_rows;
		int data_cols;
		int output_classes;
		std::string filename;
		int layer_mult;
		int in_size;
		int out_size;
		int cols;
	};

	nnet(input_parms data);
	~nnet();

	void run(const int EPOCHS, std::vector<std::vector<type_t>> data, std::vector<std::vector<type_t>> one_hot);

private:
	int DATA_ROWS;
	int DATA_COLS;
	int OUTPUT_CLASSES;
	std::string FILENAME;
	int LAYER_MULT;
	int IN_SIZE;
	int OUT_SIZE;
	int COLS;
};


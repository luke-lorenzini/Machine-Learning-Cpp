#pragma once
#define _USE_FIXED_RAND

typedef double type_t;

class neuron
{
public:
	neuron(int input, int output);
	~neuron();

	void fwd(std::vector<type_t> &x);
	void fwd(concurrency::array_view<type_t, 2> &ar_x);
	void bkwd(std::vector<type_t> &delta_in);
	void bkwd(concurrency::array_view<type_t, 2>  &ar_delta_in);
	void accm(std::vector<type_t> &x);
	void accm(concurrency::array_view<type_t, 2> &ar_x);
	void updt(int samples);
	void set_error();

	std::vector<type_t>& get_y();
	std::vector<type_t>& get_delta();
	std::vector<type_t>& get_W();
	std::vector<type_t>& get_error();
	
private:
	type_t alpha = 1;
	const int COLS = 1;

	std::random_device rd;

	int x_rows;
	int x_cols;

	int t_x_rows;
	int t_x_cols;
	int t_x_size;
	std::vector<type_t> t_x;

	int W_rows;
	int W_cols;
	int W_size;
	std::vector<type_t> W;

	int W_Trans_rows;
	int W_Trans_cols;
	int W_Trans_size;
	std::vector<type_t> W_Trans;

	int delta_in_rows;
	int delta_in_cols;

	int z_rows;
	int z_cols;
	int z_size;
	std::vector<type_t> z;

	int y_rows;
	int y_cols;
	int y_size;
	std::vector<type_t> y;

	int t_y_rows;
	int t_y_cols;
	int t_y_size;
	std::vector<type_t> t_y;

	int t_e0_rows;
	int t_e0_cols;
	int t_e0_size;
	std::vector<type_t> t_e0;

	int error_rows;
	int error_cols;
	int error_size;
	std::vector<type_t> error;

	int ones_rows;
	int ones_cols;
	int ones_size;
	std::vector<type_t> ones;

	int delta_W_rows;
	int delta_W_cols;
	int delta_W_size;
	std::vector<type_t> delta_W;

	int t_delta_W_rows;
	int t_delta_W_cols;
	int t_delta_W_size;
	std::vector<type_t> t_delta_W;

	int delta_rows;
	int delta_cols;
	int delta_size;
	std::vector<type_t> delta;

	int t_delta_out_rows;
	int t_delta_out_cols;
	int t_delta_out_size;
	std::vector<type_t> t_delta_out;

	void init_rand_real(int size, std::vector<type_t> &vect);
	void init_zeros(int size, std::vector<type_t> &vect);
	void init_ones(int size, std::vector<type_t> &vect);

	void activate(int actFunct);
	void activate_der(int actFunct);
	void updateAlpha();
};
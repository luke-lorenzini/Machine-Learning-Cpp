#pragma once

template <class type_t>
class nnet_math
{
private:
	nnet_math() {}

	static const int RANK = 2;

public:
	static void matrix_mult(const std::vector<type_t> &a, const std::vector<type_t> &b, std::vector<type_t> &res, int M, int N, int W);
	static void matrix_mult(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res);
	static void matrix_mult_tile(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res);
	static void matrix_sub(const std::vector<type_t> &a, const std::vector<type_t> &b, std::vector<type_t> &res, int M, int N);
	static void scalar_mult(const std::vector<type_t>& a, type_t mult, std::vector<type_t>& res, int M, int N);
	static void scalar_mult(concurrency::array_view<type_t, RANK>& ar_a, type_t mult, concurrency::array_view<type_t, RANK>& ar_res);
	static void scalar_div(const std::vector<type_t>& a, type_t div, std::vector<type_t>& res, int M, int N);
	static void scalar_div(concurrency::array_view<type_t, RANK>& ar_a, type_t div, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_sub(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_add(const std::vector<type_t>& a, const std::vector<type_t>& b, std::vector<type_t>& res, int M, int N);
	static void logistic(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void tanh(const std::vector<type_t>& a, std::vector<type_t>& res, int M, int N);
	static void tanh(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void relu(const std::vector<type_t>& a, std::vector<type_t>& res, int M, int N);
	static void relu(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void relu_der(const std::vector<type_t>& a, std::vector<type_t>& res, int M, int N);
	static void relu_der(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_add(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_prod(const std::vector<type_t>& a, const std::vector<type_t>& b, std::vector<type_t>& res, int M, int N);
	static void matrix_prod(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_trans(const std::vector<type_t>& a, std::vector<type_t>& res, int M, int N);
	static void matrix_trans(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void logistic(const std::vector<type_t>& a, std::vector<type_t>& res, int M, int N);
};

template<class type_t>
inline void nnet_math<type_t>::matrix_mult(const std::vector<type_t>& a, const std::vector<type_t>& b, std::vector<type_t>& res, int M, int N, int W)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<const type_t, RANK> ar_b(N, W, b);
	concurrency::array_view<type_t, RANK> ar_res(M, W, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		for (auto inner = 0; inner < RANK; inner++) {
			ar_res[idx] += ar_a(row, inner) * ar_b(inner, col);
		}
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::matrix_mult(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		for (auto inner = 0; inner < RANK; inner++)
		{
			ar_res[idx] += ar_a(row, inner) * ar_b(inner, col);
		}
	});
}

template<class type_t>
inline void nnet_math<type_t>::matrix_mult_tile(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res)
{
	parallel_for_each(ar_res.extent.tile<RANK, RANK>(), [=](concurrency::tiled_index<RANK, RANK> tidx) restrict(amp)
	{
		auto row = tidx.global[0];
		auto col = tidx.global[1];
		type_t sum = 0.0;
		for (auto inner = 0; inner < RANK; inner++)
		{
			sum += ar_a(row, inner) * ar_b(inner, col);
		}

		ar_res[tidx] = sum;
	});
}

template<class type_t>
inline void nnet_math<type_t>::matrix_sub(const std::vector<type_t>& a, const std::vector<type_t>& b, std::vector<type_t>& res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<const type_t, RANK> ar_b(M, N, b);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] - ar_b[row][col];
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::scalar_mult(const std::vector<type_t> &a, type_t mult, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] * mult;
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::scalar_mult(concurrency::array_view<type_t, RANK> &ar_a, type_t mult, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] * mult;
	});
}

template<class type_t>
inline void nnet_math<type_t>::scalar_div(const std::vector<type_t> &a, type_t div, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] / div;
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::scalar_div(concurrency::array_view<type_t, RANK> &ar_a, type_t div, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] / div;
	});
}

template<class type_t>
inline void nnet_math<type_t>::matrix_sub(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] - ar_b[row][col];
	});
}

template<class type_t>
inline void nnet_math<type_t>::matrix_add(const std::vector<type_t> &a, const std::vector<type_t> &b, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<const type_t, RANK> ar_b(M, N, b);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] + ar_b[row][col];
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::matrix_add(concurrency::array_view< type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] + ar_b[row][col];
	});
}

template<class type_t>
inline void nnet_math<type_t>::matrix_prod(const std::vector<type_t> &a, const std::vector<type_t> &b, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<const type_t, RANK> ar_b(M, N, b);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] * ar_b[row][col];
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::matrix_prod(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] * ar_b[row][col];
	});
}

template<class type_t>
inline void nnet_math<type_t>::matrix_trans(const std::vector<type_t> &a, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	/* Dimensions intentionally changed for transpose */
	concurrency::array_view<type_t, RANK> ar_res(N, M, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[col][row];
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::matrix_trans(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[col][row];
	});
}

template<class type_t>
inline void nnet_math<type_t>::logistic(const std::vector<type_t> &a, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = 1 / (1 + concurrency::precise_math::exp(-1 * ar_a[row][col]));
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::logistic(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = 1 / (1 + concurrency::precise_math::exp(-1 * ar_a[row][col]));
	});
}

template<class type_t>
inline void nnet_math<type_t>::tanh(const std::vector<type_t> &a, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = concurrency::precise_math::tanh(ar_a[row][col]);
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::tanh(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = concurrency::precise_math::tanh(ar_a[row][col]);
	});
}

template<class type_t>
inline void nnet_math<type_t>::relu(const std::vector<type_t> &a, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		if (ar_a[row][col] < 0)
		{
			ar_res[row][col] = 0;
		}
		else
		{
			ar_res[row][col] = ar_a[row][col];
		}
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::relu(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		if (ar_a[row][col] < 0)
		{
			ar_res[row][col] = 0;
		}
		else
		{
			ar_res[row][col] = ar_a[row][col];
		}
	});
}

template<class type_t>
inline void nnet_math<type_t>::relu_der(const std::vector<type_t> &a, std::vector<type_t> &res, int M, int N)
{
	concurrency::array_view<const type_t, RANK> ar_a(M, N, a);
	concurrency::array_view<type_t, RANK> ar_res(M, N, res);
	//ar_res.discard_data();

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		if (ar_a[row][col] < 0)
		{
			ar_res[row][col] = 0;
		}
		else
		{
			ar_res[row][col] = 1;
		}
	});

	ar_res.synchronize();
}

template<class type_t>
inline void nnet_math<type_t>::relu_der(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		if (ar_a[row][col] < 0)
		{
			ar_res[row][col] = 0;
		}
		else
		{
			ar_res[row][col] = 1;
		}
	});
}

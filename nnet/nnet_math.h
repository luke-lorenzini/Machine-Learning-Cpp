// Copyright (c) 2018 Luke Lorenzini, https://www.zinisolutions.com/
// This file is licensed under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once

//#define DISCARD
#define SYNC

template <class type_t>
class nnet_math
{
private:
	nnet_math() {}

	static const int RANK = 2;

public:
	static void matrix_mult(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res);
	static void matrix_mult_tile(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res);
	static void scalar_mult(concurrency::array_view<type_t, RANK>& ar_a, type_t mult, concurrency::array_view<type_t, RANK>& ar_res);
	static void scalar_div(concurrency::array_view<type_t, RANK>& ar_a, type_t div, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_sub(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res);
	static void logistic(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void tanh(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void relu(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void relu_der(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_add(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_prod(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res);
	static void matrix_trans(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_res);
	static void exponent(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res, type_t sum);
	static void softmax(concurrency::array_view<type_t, RANK> &ar_a, type_t sum, concurrency::array_view<type_t, RANK> &ar_res);
};

template<class type_t>
inline void nnet_math<type_t>::matrix_mult(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif

	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];
		type_t sum = 0.0;

		for (auto inner = 0; inner < ar_a.extent[1]; inner++)
		{
			sum += ar_a(row, inner) * ar_b(inner, col);
			//ar_res[idx] += ar_a(row, inner) * ar_b(inner, col);
		}
		ar_res[idx] = sum;
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::matrix_mult_tile(concurrency::array_view<type_t, RANK>& ar_a, concurrency::array_view<type_t, RANK>& ar_b, concurrency::array_view<type_t, RANK>& ar_res)
{
	static const int TileSize = 1;

#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent.tile<TileSize, TileSize>(), [=](concurrency::tiled_index<TileSize, TileSize> tidx) restrict(amp)
	{
		auto row = tidx.global[0];
		auto col = tidx.global[1];
		type_t sum = 0.0;

		for (auto i = 0; i < ar_a.extent[1]; i += TileSize)
		{
			tile_static type_t sA[TileSize][TileSize];
			tile_static type_t sB[TileSize][TileSize];
			sA[row][col] = ar_a(tidx.global[0], col + i);
			sB[row][col] = ar_b(row + i, tidx.global[1]);

			tidx.barrier.wait();

			for (auto k = 0; k < TileSize; k++)
			{
				sum += sA[row][k] * sB[k][col];
			}

			tidx.barrier.wait();
		}

		ar_res[tidx.global] = sum;
	});
#ifdef SYNC
	ar_res.synchronize();
#endif	
}

template<class type_t>
inline void nnet_math<type_t>::scalar_mult(concurrency::array_view<type_t, RANK> &ar_a, type_t mult, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] * mult;
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::scalar_div(concurrency::array_view<type_t, RANK> &ar_a, type_t div, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] / div;
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::matrix_sub(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] - ar_b[row][col];
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::matrix_add(concurrency::array_view< type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] + ar_b[row][col];
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::matrix_prod(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_b, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] * ar_b[row][col];
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::matrix_trans(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[col][row];
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::logistic(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = 1 / (1 + concurrency::precise_math::exp(-1 * ar_a[row][col]));
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::tanh(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = concurrency::precise_math::tanh(ar_a[row][col]);
	});
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::relu(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
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
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::relu_der(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
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
#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::exponent(concurrency::array_view<type_t, RANK> &ar_a, concurrency::array_view<type_t, RANK> &ar_res, type_t sum)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = concurrency::precise_math::exp(ar_a[row][col] + sum);
	});

#ifdef SYNC
	ar_res.synchronize();
#endif
}

template<class type_t>
inline void nnet_math<type_t>::softmax(concurrency::array_view<type_t, RANK> &ar_a, type_t sum, concurrency::array_view<type_t, RANK> &ar_res)
{
#ifdef DISCARD
	ar_res.discard_data();
#endif
	parallel_for_each(ar_res.extent, [=](concurrency::index<RANK> idx) restrict(amp)
	{
		auto row = idx[0];
		auto col = idx[1];

		ar_res[row][col] = ar_a[row][col] / sum;
	});	

#ifdef SYNC
	ar_res.synchronize();
#endif
}


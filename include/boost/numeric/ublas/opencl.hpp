#ifndef OPENCL_MATRIX
#define OPENCL_MATRIX

#include <sys/types.h>
#include <boost/compute/algorithm.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <iostream>

/// Include the clBLAS header. It includes the appropriate OpenCL headers
#include <clBLAS.h>


namespace opencl
{
	namespace compute = boost::compute;
	namespace ublas = boost::numeric::ublas;



	class opencl_device
	{
	private:
		compute::device device;
		compute::context context;
		compute::command_queue queue;

	public:
		opencl_device() {}

		opencl_device(unsigned int device_number)
		{
			this->device = compute::system::devices().at(device_number);
			this->context = compute::context(device);
			this->queue = compute::command_queue(context, device);
		}


		void initialize(unsigned int device_number)
		{
			this->device = compute::system::devices().at(device_number);
			this->context = compute::context(device);
			this->queue = compute::command_queue(context, device);
		}

		compute::device getDevice()
		{
			return device;
		}

		compute::context getContext()
		{
			return context;
		}

		compute::command_queue getQueue()
		{
			return queue;
		}

	};

	inline void startOpencl()
	{
		///start clBlas
		clblasSetup();

	}

	inline void endOpencl()
	{
		/// Finalize clBLAS

		clblasTeardown();

	}

	///copy data from ublas matrix to gpu vector
	template <class T, class F, class A>
	void copy_to_gpu(ublas::matrix<T, F, A>& m, compute::vector<T>& v, compute::command_queue& queue)
	{
		compute::copy(
			m.data().begin(),
			m.data().end(),
			v.begin(),
			queue
		);

	}

	///copy data from gpu vector to ublas matrix
	template <class T, class F, class A>
	void copy_from_gpu(ublas::matrix<T, F, A>& m, compute::vector<T>& v, compute::command_queue& queue)
	{
		compute::copy(
			v.begin(),
			v.end(),
			m.data().begin(),
			queue
		);

	}



#define ONE_DOUBLE_COMPLEX  { { 1.0, 00.0 } }
#define ONE_FLOAT_COMPLEX  { { 1.0f, 00.0f } }


	/**
	This function computes the prodect of 2 matrices (A*B) and stores it at matrix result
	it first transfers the data of matrix a,b to the device and execute a clBlas kernel according to
	the data type of the matrices to multiply them
	*/
	template <class T, class F, class A>
	BOOST_UBLAS_INLINE
		void prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, ublas::matrix<T, F, A>& result, opencl_device& cl_device)
	{


		///get data from device
		compute::device device = cl_device.getDevice();
		compute::context context = cl_device.getContext();
		compute::command_queue queue = cl_device.getQueue();



		///on the GPU to move the data of matrix (a) to
		compute::vector<T> aHolder(a.size1() * a.size2(), context);

		///on the GPU to move the data of matrix (b) to
		compute::vector<T> bHolder(b.size1() * b.size2(), context);

		///on the GPU to hold the result 
		compute::vector<T> resultHolder(a.size1() * b.size2(), 0, queue);


		///copy the data from a to aHolder
		copy_to_gpu(a, aHolder, queue);

		///copy data from b to bHolder
		copy_to_gpu(b, bHolder, queue);



		result.resize(a.size1(), b.size2());

		cl_int err;
		cl_event event = NULL;

		clblasOrder Order = std::is_same<F, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
		int lda = Order == clblasRowMajor ? a.size2() : a.size1();
		int ldb = Order == clblasRowMajor ? b.size2() : a.size2();
		int ldc = Order == clblasRowMajor ? b.size2() : a.size1();



		if (std::is_same<T, float>::value)
			///Call clBLAS extended function. Perform gemm for float
			err = clblasSgemm(Order, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				1, (cl_mem)aHolder.begin().get_buffer().get(), 0, lda,
				(cl_mem)bHolder.begin().get_buffer().get(), 0, ldb, 1,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, ldc,
				1, &(queue.get()), 0, NULL, &event);


		else if (std::is_same<T, double>::value)
			///Call clBLAS extended function. Perform gemm for double
			err = clblasDgemm(Order, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				1, (cl_mem)aHolder.begin().get_buffer().get(), 0, lda,
				(cl_mem)bHolder.begin().get_buffer().get(), 0, ldb, 1,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, ldc,
				1, &(queue.get()), 0, NULL, &event);

		else if (std::is_same<T, std::complex<float>>::value)
			///Call clBLAS extended function. Perform gemm for double
			err = clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				ONE_FLOAT_COMPLEX, (cl_mem)aHolder.begin().get_buffer().get(), 0, lda,
				(cl_mem)bHolder.begin().get_buffer().get(), 0, ldb, ONE_FLOAT_COMPLEX,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, ldc,
				1, &(queue.get()), 0, NULL, &event);

		else if (std::is_same<T, std::complex<double>>::value)
			///Call clBLAS extended function. Perform gemm for double
			err = clblasZgemm(Order, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				ONE_DOUBLE_COMPLEX, (cl_mem)aHolder.begin().get_buffer().get(), 0, lda,
				(cl_mem)bHolder.begin().get_buffer().get(), 0, ldb, ONE_DOUBLE_COMPLEX,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, ldc,
				1, &(queue.get()), 0, NULL, &event);



		///Wait for calculations to be finished.
		clWaitForEvents(1, &event);



		copy_from_gpu(result, resultHolder, queue);


	}


	/**
	This function computes the prodect of 2 matrices (A*B) and returns the result
	it uses the function prod(A,B,result) and returns result
	*/

	template <class T, class F, class A>
	BOOST_UBLAS_INLINE
		ublas::matrix<T, F, A> prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, opencl_device cl_device)
	{
		ublas::matrix<T, F, A> result;
		prod(a, b, result, cl_device);
		return result;
	}



}


#endif 
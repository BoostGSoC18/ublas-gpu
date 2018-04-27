#ifndef OPENCL_MATRIX
#define OPENCL_MATRIX

#include <sys/types.h>
#include <boost/compute/algorithm.hpp>
#include <boost/numeric/ublas/matrix.hpp>
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
	template <class T>
	void copy_to_gpu(ublas::matrix<T>& m, compute::vector<T>& v, compute::command_queue& queue)
	{
		compute::copy(
			m.data().begin(),
			m.data().end(),
			v.begin(),
			queue
		);

	}

	///copy data from gpu vector to ublas matrix
	template <class T>
	void copy_from_gpu(ublas::matrix<T>& m, compute::vector<T>& v, compute::command_queue& queue)
	{
		compute::copy(
			v.begin(),
			v.end(),
			m.data().begin(),
			queue
		);

	}

	/**
	This function computes the prodect of 2 matrices (A*B) and stores it at matrix result
	it first transfers the data of matrix a,b to the device and execute a clBlas kernel according to
	the data type of the matrices to multiply them
	*/
	template <class T>
	BOOST_UBLAS_INLINE
		void prod(ublas::matrix<T>& a, ublas::matrix<T>& b, ublas::matrix<T>& result, opencl_device& cl_device)
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

		if (std::is_same<T, float>::value)
			///Call clBLAS extended function. Perform gemm for float
			err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				1, (cl_mem)aHolder.begin().get_buffer().get(), 0, a.size2(),
				(cl_mem)bHolder.begin().get_buffer().get(), 0, b.size2(), 1,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, b.size2(),
				1, &(queue.get()), 0, NULL, &event);


		else if (std::is_same<T, double>::value)
			///Call clBLAS extended function. Perform gemm for double
			err = clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				1, (cl_mem)aHolder.begin().get_buffer().get(), 0, a.size2(),
				(cl_mem)bHolder.begin().get_buffer().get(), 0, b.size2(), 1,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, b.size2(),
				1, &(queue.get()), 0, NULL, &event);



		///Wait for calculations to be finished.
		clWaitForEvents(1, &event);



		copy_from_gpu(result, resultHolder, queue);


	}


	/**
	This function computes the prodect of 2 matrices (A*B) and returns the result
	it uses the function prod(A,B,result) and returns result
	*/

	template <class T>
	BOOST_UBLAS_INLINE
		ublas::matrix<T> prod(ublas::matrix<T>& a, ublas::matrix<T>& b, opencl_device cl_device)
	{
		ublas::matrix<T> result;
		prod(a, b, result, cl_device);
		return result;
	}



}


#endif 
#ifndef OPENCL_MATRIX
#define OPENCL_MATRIX

#include <sys/types.h>
#include <boost\compute\algorithm.hpp>
#include <boost\numeric\ublas\matrix.hpp>

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
			this->device = compute::system::devices().at(DEVICE_NUMBER);
			this->context = compute::context(device);
			this->queue = compute::command_queue(context, device);
		}


		void initialize(unsigned int device_number)
		{
			this->device = compute::system::devices().at(DEVICE_NUMBER); 
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


	/**
	This function computes the prodect of 2 matrices (A*B) and stores it at matrix result
	it first transfers the data of matrix a,b to the device and execute a clBlas kernel according to
	the data type of the matrices to multiply them
	*/
	template <class T>
	BOOST_UBLAS_INLINE
		void prod(ublas::matrix<T>& a, ublas::matrix<T>& b, ublas::matrix<T>& result , opencl_device cl_device)
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
		compute::vector<T> resultHolder(a.size1() * b.size2(), context);


		///copy the data from a to aHolder
		compute::copy(
			a.data().begin(),
			a.data().end(),
			aHolder.begin(),
			queue
		);

		///copy data from b to bHolder
		compute::copy(
			b.data().begin(),
			b.data().end(),
			bHolder.begin(),
			queue
		);


		cl_event event = NULL;

		if (std::is_same<T, float>::value)
			///Call clBLAS extended function. Perform gemm for float
			clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				1, (cl_mem)aHolder.begin().get_buffer().get(), 0, a.size2(),
				(cl_mem)bHolder.begin().get_buffer().get(), 0, b.size2(), 1,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, b.size2(),
				1, &(queue.get()), 0, NULL, &event);


		else if (std::is_same<T, double>::value)
			///Call clBLAS extended function. Perform gemm for double
			clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				1, (cl_mem)aHolder.begin().get_buffer().get(), 0, a.size2(),
				(cl_mem)bHolder.begin().get_buffer().get(), 0, b.size2(), 1,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, b.size2(),
				1, &(queue.get()), 0, NULL, &event);





		///Wait for calculations to be finished.
		clWaitForEvents(1, &event);

		result.resize(a.size1(), b.size2());

		compute::copy(
			resultHolder.begin(),
			resultHolder.end(),
			result.data().begin(),
			queue
		);



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


	template <class T>
	BOOST_UBLAS_INLINE
		void prod(ublas::matrix_expression<T>& expressionA, ublas::matrix_expression<T>& expressionB, ublas::matrix<T>& result, opencl_device cl_device)
	{
		ublas::matrix<T> a = expressionA;
		ublas::matrix<T> b = expressionB;

		prod(a, b, result, cl_device);
	}



	template <class T>
	BOOST_UBLAS_INLINE
		ublas::matrix<T> prod(ublas::matrix_expression<T>& expressionA, ublas::matrix_expression<T>& expressionB, opencl_device cl_device)
	{
		ublas::matrix<T> result;
		prod(expressionA, expressionB, result, cl_device);

		return result;
	}

}


#endif 
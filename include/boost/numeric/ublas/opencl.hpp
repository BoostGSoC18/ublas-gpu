#ifndef OPENCL_MATRIX
#define OPENCL_MATRIX

#include <sys/types.h>
#include <boost\compute\algorithm.hpp>
#include <boost\numeric\ublas\matrix.hpp>

/* Include the clBLAS header. It includes the appropriate OpenCL headers */
#include <clBLAS.h>

namespace opencl
{
	namespace compute = boost::compute;
	namespace ublas = boost::numeric::ublas;

	// for setting up the boost.compute library 
	compute::device device;
	compute::context context;
	compute::command_queue queue;


	void startOpencl(int deviceNumber)
	{
		device = compute::system::devices().at(deviceNumber);
		context = compute::context(device);
		queue = compute::command_queue(context, device);

		//start clBlas
		clblasSetup();
		
	}

	void endOpencl()
	{
		/* Finalize clBLAS */

		clblasTeardown();

	}

	template <class T>
	void prod(ublas::matrix<T>& a, ublas::matrix<T>& b, ublas::matrix<T>& result)
	{

		cl_int err;

		//on the GPU to move the data of matrix (a) to
		compute::vector<T> aHolder(a.size1() * a.size2(), context);

		//on the GPU to move the data of matrix (b) to
		compute::vector<T> bHolder(b.size1() * b.size2(), context);

		//on the GPU to hold the result 
		compute::vector<T> resultHolder(a.size1() * b.size2(), context);

		
		//copy the data from a to aHolder
		compute::copy(
			a.data().begin(),
			a.data().end(),
			aHolder.begin(),
			queue
		);

		//copy data from b to bHolder
		compute::copy(
			b.data().begin(),
			b.data().end(),
			bHolder.begin(),
			queue
		);


		cl_event event = NULL;

		if (std::is_same<T, float>::value)
		/* Call clBLAS extended function. Perform gemm for float */
		err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
			a.size1(), b.size2(), a.size2(),
			1, (cl_mem)aHolder.begin().get_buffer().get(), 0, a.size2(),
			(cl_mem)bHolder.begin().get_buffer().get(), 0, b.size2(), 1,
			(cl_mem)resultHolder.begin().get_buffer().get(), 0, b.size2(),
			1, &(queue.get()), 0, NULL, &event);


		else if (std::is_same<T, double>::value)
			/* Call clBLAS extended function. Perform gemm for double */
			err = clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
				a.size1(), b.size2(), a.size2(),
				1, (cl_mem)aHolder.begin().get_buffer().get(), 0, a.size2(),
				(cl_mem)bHolder.begin().get_buffer().get(), 0, b.size2(), 1,
				(cl_mem)resultHolder.begin().get_buffer().get(), 0, b.size2(),
				1, &(queue.get()), 0, NULL, &event);

		



		/* Wait for calculations to be finished. */
		err = clWaitForEvents(1, &event);
		queue.finish();


		compute::copy(
			resultHolder.begin(),
			resultHolder.end(),
			result.data().begin(),
			queue
		);



	}

	

	template <class T>
	ublas::matrix<T> prod(ublas::matrix<T>& a, ublas::matrix<T>& b)
	{
		ublas::matrix<T> result(a.size1(), b.size2());
		prod(a, b, result);
		return result;
	}

}


#endif 
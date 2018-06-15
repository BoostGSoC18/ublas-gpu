#ifndef OPENCL_OPERATIONS
#define OPENCL_OPERATIONS

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/compute/buffer.hpp>

/// Include the clBLAS header. It includes the appropriate OpenCL headers
#include <clBLAS.h>

namespace boost {
  namespace numeric {
	namespace ublas {


	  namespace opencl
	  {
		namespace compute = boost::compute;
		namespace ublas = boost::numeric::ublas;

	    #define ONE_DOUBLE_COMPLEX  { { 1.0, 00.0 } }
	    #define ONE_FLOAT_COMPLEX  { { 1.0f, 00.0f } }

		/**This function computes the product of 2 matrices (a*b) and stores it at matrix result all 3 matrices are on device
		*
		* a and b are originally on device (on the same device) and the result is left on the same device.
		*
		* \param a matrix A of the product (A*B) that is on opencl_device device
		* \param b matrix B of the product (A*B) that is on the opencl_device device
		* \param result matrix on device to store the product of the result of (A*B)
		*
		* \tparam T datatype of the matrices
		* \tparam L layout of the matrices (row_majot or column_major)
		*/
		template <class T, class F>
		BOOST_UBLAS_INLINE
		  void prod(ublas::matrix<T, F, opencl::storage>& a, ublas::matrix<T, F, opencl::storage>& b, ublas::matrix<T, F, opencl::storage>& result)
		{


		  //get data from device
		  compute::device device = a.device().getDevice();
		  compute::context context = a.device().getContext();
		  compute::command_queue queue = a.device().getQueue();


		  result.resize(a.size1(), b.size2());
		  result.fill(0);

		  cl_event event = NULL;

		  clblasOrder Order = std::is_same<F, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
		  int lda = Order == clblasRowMajor ? a.size2() : a.size1();
		  int ldb = Order == clblasRowMajor ? b.size2() : a.size2();
		  int ldc = Order == clblasRowMajor ? b.size2() : a.size1();



		  if (std::is_same<T, float>::value)
			//Call clBLAS extended function. Perform gemm for float
			clblasSgemm(Order, clblasNoTrans, clblasNoTrans,
			  a.size1(), b.size2(), a.size2(),
			  1, (cl_mem)a.data().begin().get_buffer().get(), 0, lda,
			  (cl_mem)b.data().begin().get_buffer().get(), 0, ldb, 1,
			  (cl_mem)result.data().begin().get_buffer().get(), 0, ldc,
			  1, &(queue.get()), 0, NULL, &event);


		  else if (std::is_same<T, double>::value)
			//Call clBLAS extended function. Perform gemm for double
			clblasDgemm(Order, clblasNoTrans, clblasNoTrans,
			  a.size1(), b.size2(), a.size2(),
			  1, (cl_mem)a.data().begin().get_buffer().get(), 0, lda,
			  (cl_mem)b.data().begin().get_buffer().get(), 0, ldb, 1,
			  (cl_mem)result.data().begin().get_buffer().get(), 0, ldc,
			  1, &(queue.get()), 0, NULL, &event);

		  else if (std::is_same<T, std::complex<float>>::value)
			//Call clBLAS extended function. Perform gemm for double
			clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
			  a.size1(), b.size2(), a.size2(),
			  ONE_FLOAT_COMPLEX, (cl_mem)a.data().begin().get_buffer().get(), 0, lda,
			  (cl_mem)b.data().begin().get_buffer().get(), 0, ldb, ONE_FLOAT_COMPLEX,
			  (cl_mem)result.data().begin().get_buffer().get(), 0, ldc,
			  1, &(queue.get()), 0, NULL, &event);

		  else if (std::is_same<T, std::complex<double>>::value)
			//Call clBLAS extended function. Perform gemm for double
			clblasZgemm(Order, clblasNoTrans, clblasNoTrans,
			  a.size1(), b.size2(), a.size2(),
			  ONE_DOUBLE_COMPLEX, (cl_mem)a.data().begin().get_buffer().get(), 0, lda,
			  (cl_mem)b.data().begin().get_buffer().get(), 0, ldb, ONE_DOUBLE_COMPLEX,
			  (cl_mem)result.data().begin().get_buffer().get(), 0, ldc,
			  1, &(queue.get()), 0, NULL, &event);



		  //Wait for calculations to be finished.
		  clWaitForEvents(1, &event);



		}




		
		/**This function computes the product of 2 matrices not on device (a*b) and stores it at matrix result which is also not on device
		*
		* a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to matrix result
		*
		* \param a matrix A of the product (A*B) that is not on opencl_device device
		* \param b matrix B of the product (A*B) that is not on the opencl_device device
		* \param result matrix on device to store the product of the result of (A*B)
		* \param device has the information about the device which as the two matrices and which will do the computation
		*
		* \tparam T datatype of the matrices
		* \tparam L layout of the matrices (row_majot or column_major)
		* \tparam A storage type that has the data of the matrices
		*/
		template <class T, class F, class A>
		BOOST_UBLAS_INLINE
		  void prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, ublas::matrix<T, F, A>& result, opencl_device& device)
		{

		  ///copy the data from a to aHolder
		  ublas::matrix<T, F, opencl::storage> aHolder(device);
		  aHolder.to_host(a);

		  ///copy the data from b to bHolder
		  ublas::matrix<T, F, opencl::storage> bHolder(device);
		  bHolder.to_host(b);

		  ublas::matrix<T, F, opencl::storage> resultHolder(device);

		  prod(aHolder, bHolder, resultHolder); //call the prod function that multiplies a function already on gpu

		  resultHolder.from_host(result);


		}


		/**This function computes the product of 2 matrices not on device (a*b) and stores it at matrix result which is also not on device
		*
		* a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device returned
		*
		* \param a matrix A of the product (A*B) that is not on opencl_device device
		* \param b matrix B of the product (A*B) that is not on the opencl_device device
		* \param device has the information about the device which as the two matrices and which will do the computation
		*
		* \tparam T datatype of the matrices
		* \tparam L layout of the matrices (row_majot or column_major)
		* \tparam A storage type that has the data of the matrices
		*/

		template <class T, class F, class A>
		BOOST_UBLAS_INLINE
		  ublas::matrix<T, F, A> prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, opencl_device device)
		{
		  ublas::matrix<T, F, A> result;
		  prod(a, b, result, device);
		  return result;
		}



	  }//opencl

	}//ublas
  }//numeric
}//boost


#endif 
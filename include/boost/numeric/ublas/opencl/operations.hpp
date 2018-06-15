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
* \param a matrix A of the product (A*B) that is on device
* \param b matrix B of the product (A*B) that is on the device
* \param result matrix on device to store the product of the result of (A*B)
*
* \tparam T datatype of the matrices
* \tparam L layout of the matrices (row_majot or column_major)
*/
template <class T, class F>
  void prod(ublas::matrix<T, F, opencl::storage>& a, ublas::matrix<T, F, opencl::storage>& b, ublas::matrix<T, F, opencl::storage>& result)
{
  //check all matrices are on same context
  assert((a.queue().get_context() == b.queue().get_context()) && (a.queue().get_context() == result.queue().get_context()));

  //get data from device
  compute::device device = a.queue().get_device();
  compute::context context = a.queue().get_context();
  compute::command_queue queue = a.queue();

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
	  1, (cl_mem)a.begin().get_buffer().get(), 0, lda,
	  (cl_mem)b.begin().get_buffer().get(), 0, ldb, 1,
	  (cl_mem)result.begin().get_buffer().get(), 0, ldc,
	  1, &(queue.get()), 0, NULL, &event);


  else if (std::is_same<T, double>::value)
	//Call clBLAS extended function. Perform gemm for double
	clblasDgemm(Order, clblasNoTrans, clblasNoTrans,
	  a.size1(), b.size2(), a.size2(),
	  1, (cl_mem)a.begin().get_buffer().get(), 0, lda,
	  (cl_mem)b.begin().get_buffer().get(), 0, ldb, 1,
	  (cl_mem)result.begin().get_buffer().get(), 0, ldc,
	  1, &(queue.get()), 0, NULL, &event);

  else if (std::is_same<T, std::complex<float>>::value)
	//Call clBLAS extended function. Perform gemm for double
	clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
	  a.size1(), b.size2(), a.size2(),
	  ONE_FLOAT_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
	  (cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_FLOAT_COMPLEX,
	  (cl_mem)result.begin().get_buffer().get(), 0, ldc,
	  1, &(queue.get()), 0, NULL, &event);

  else if (std::is_same<T, std::complex<double>>::value)
	//Call clBLAS extended function. Perform gemm for double
	clblasZgemm(Order, clblasNoTrans, clblasNoTrans,
	  a.size1(), b.size2(), a.size2(),
	  ONE_DOUBLE_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
	  (cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_DOUBLE_COMPLEX,
	  (cl_mem)result.begin().get_buffer().get(), 0, ldc,
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
* \param queue has the queue of the device which has the result matrix and which will do the computation
*
* \tparam T datatype of the matrices
* \tparam L layout of the matrices (row_majot or column_major)
* \tparam A storage type that has the data of the matrices
*/
template <class T, class F, class A>
  void prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, ublas::matrix<T, F, A>& result, compute::command_queue &queue)
{

  ///copy the data from a to aHolder
  ublas::matrix<T, F, opencl::storage> aHolder(a.size1(), a.size2(), queue);
  aHolder.to_host(a);

  ///copy the data from b to bHolder
  ublas::matrix<T, F, opencl::storage> bHolder(b.size1(), b.size2() ,queue);
  bHolder.to_host(b);

  ublas::matrix<T, F, opencl::storage> resultHolder(a.size1(), b.size2(), queue);

  prod(aHolder, bHolder, resultHolder); //call the prod function that multiplies a function already on gpu

  resultHolder.from_host(result);


}


/**This function computes the product of 2 matrices not on device (a*b) and stores it at matrix result which is also not on device
*
* a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device returned
*
* \param a matrix A of the product (A*B) that is not on opencl_device device
* \param b matrix B of the product (A*B) that is not on the opencl_device device
* \param queue has the queue of the device which will do the computation
*
* \tparam T datatype of the matrices
* \tparam L layout of the matrices (row_majot or column_major)
* \tparam A storage type that has the data of the matrices
*/

template <class T, class F, class A>
  ublas::matrix<T, F, A> prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, compute::command_queue &queue)
{
  ublas::matrix<T, F, A> result(a.size1(), b.size2());
  prod(a, b, result, queue);
  return result;
}



}//opencl

}//ublas
}//numeric
}//boost


#endif 
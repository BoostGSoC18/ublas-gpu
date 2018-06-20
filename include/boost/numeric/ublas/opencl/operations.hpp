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

//Matrix-Matrix multiplication

/**This function computes the product of 2 matrices (a*b) and stores it at matrix result all 3 matrices are on device
*
* a and b are originally on device (on the same device) and the result is left on the same device.
*
* \param a matrix A of the product (A*B) that is on device
* \param b matrix B of the product (A*B) that is on the device
* \param result matrix on device to store the product of the result of (A*B)
* \param queue has the queue of the device which has the result matrix and which will do the computation
*
* \tparam T datatype of the matrices
* \tparam L layout of the matrices (row_major or column_major)
*/
template <class T, class F>
  void prod(ublas::matrix<T, F, opencl::storage>& a, ublas::matrix<T, F, opencl::storage>& b, ublas::matrix<T, F, opencl::storage>& result , compute::command_queue & queue)
{
  //check all matrices are on same context
  assert(  (a.device() == b.device()) && (a.device() == result.device()) && (a.device()== queue.get_device()) );

  //check dimension of matrices (MxN) * (NxK)
  assert(a.size2() == b.size1());

  result.fill(0, queue);

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
	//Call clBLAS extended function. Perform gemm for complext float
	clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
	  a.size1(), b.size2(), a.size2(),
	  ONE_FLOAT_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
	  (cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_FLOAT_COMPLEX,
	  (cl_mem)result.begin().get_buffer().get(), 0, ldc,
	  1, &(queue.get()), 0, NULL, &event);

  else if (std::is_same<T, std::complex<double>>::value)
	//Call clBLAS extended function. Perform gemm for complex double
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
* \param a matrix A of the product (A*B) that is not on device
* \param b matrix B of the product (A*B) that is not on the device
* \param result matrix on device to store the product of the result of (A*B)
* \param queue has the queue of the device which has the result matrix and which will do the computation
*
* \tparam T datatype of the matrices
* \tparam L layout of the matrices (row_major or column_major)
* \tparam A storage type that has the data of the matrices
*/
template <class T, class F, class A>
  void prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, ublas::matrix<T, F, A>& result, compute::command_queue &queue)
{

  ///copy the data from a to aHolder
  ublas::matrix<T, F, opencl::storage> aHolder(a.size1(), a.size2(), queue.get_context());
  aHolder.from_host(a,queue);

  ///copy the data from b to bHolder
  ublas::matrix<T, F, opencl::storage> bHolder(b.size1(), b.size2() ,queue.get_context());
  bHolder.from_host(b,queue);

  ublas::matrix<T, F, opencl::storage> resultHolder(a.size1(), b.size2(), queue.get_context());

  prod(aHolder, bHolder, resultHolder, queue); //call the prod function that multiplies a function already on gpu

  resultHolder.to_host(result,queue);


}


/**This function computes the product of 2 matrices not on device (a*b) and stores it at matrix result which is also not on device
*
* a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
*
* \param a matrix A of the product (A*B) that is not on device (it's on the host)
* \param b matrix B of the product (A*B) that is not on the device (it's on the host)
* \param queue has the queue of the device which has the result matrix and which will do the computation
*
* \tparam T datatype of the matrices
* \tparam L layout of the matrices (row_major or column_major)
* \tparam A storage type that has the data of the matrices
*/

template <class T, class F, class A>
  ublas::matrix<T, F, A> prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, compute::command_queue &queue)
{
  ublas::matrix<T, F, A> result(a.size1(), b.size2());
  prod(a, b, result, queue);
  return result;
}





  //Matrix-vector multiplication

  /**This function computes the product of matrix * vector (a*b) and stores it at vactor result all 3 are on same device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a matrix A of the product (A*B) that is on device
  * \param b vectoe B of the product (A*B) that is on the device
  * \param result vector on device to store the product of the result of (A*B)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the data
  * \tparam L layout of the matrix (row_major or column_major)
  */
  template <class T, class F>
  void prod(ublas::matrix<T, F, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue & queue)
  {
	//check all matrices are on same context
	assert((a.device() == b.device()) && (a.device() == result.device()) && (a.device() == queue.get_device()));


	//check dimension of matricx and vector (MxN) * (Nx1)
	assert(a.size2() == b.size());


	result.fill(0, queue);

	cl_event event = NULL;

	clblasOrder Order = std::is_same<F, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
	int lda = Order == clblasRowMajor ? a.size2() : a.size1();
	int ldb = Order == clblasRowMajor ? 1 : a.size2();
	int ldc = Order == clblasRowMajor ? 1 : a.size1();




	if (std::is_same<T, float>::value)
	  //Call clBLAS extended function. Perform gemm for float
	  clblasSgemm(Order, clblasNoTrans, clblasNoTrans,
		a.size1(), 1, a.size2(),
		1, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, 1,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);


	else if (std::is_same<T, double>::value)
	  //Call clBLAS extended function. Perform gemm for double
	  clblasDgemm(Order, clblasNoTrans, clblasNoTrans,
		a.size1(), 1, a.size2(),
		1, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, 1,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);

	else if (std::is_same<T, std::complex<float>>::value)
	  //Call clBLAS extended function. Perform gemm for complex float
	  clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
		a.size1(), 1, a.size2(),
		ONE_FLOAT_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_FLOAT_COMPLEX,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);

	else if (std::is_same<T, std::complex<double>>::value)
	  //Call clBLAS extended function. Perform gemm for complex double
	  clblasZgemm(Order, clblasNoTrans, clblasNoTrans,
		a.size1(), 1, a.size2(),
		ONE_DOUBLE_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_DOUBLE_COMPLEX,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);



	//Wait for calculations to be finished.
	clWaitForEvents(1, &event);



  }


  /**This function computes the product of matrix*vector not on device (a*b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to matrix result
  *
  * \param a matrix A of the product (A*B) that is not on device
  * \param b vactor B of the product (A*B) that is not on the device
  * \param result matrix on device to store the product of the result of (A*B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrix and the vector
  */
  template <class T, class F, class A>
  void prod(ublas::matrix<T, F, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::matrix<T, F, opencl::storage> aHolder(a.size1(), a.size2(), queue.get_context());
	aHolder.from_host(a, queue);

	///copy the data from b to bHolder
	ublas::vector<T, opencl::storage> bHolder(b.size(), queue.get_context());
	bHolder.from_host(b, queue);

	ublas::vector<T, opencl::storage> resultHolder(a.size1(), queue.get_context());

	prod(aHolder, bHolder, resultHolder, queue); //call the prod function that multiplies a function already on gpu

	resultHolder.to_host(result, queue);


  }


  /**This function computes the product of matrix*vector not on device (a*b) and retirns result vector
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a matrix A of the product (A*B) that is not on device (it's on host)
  * \param b vector B of the product (A*B) that is not on device (it's on host)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype
  * \tparam L layout of the matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrix and the vector
  */

  template <class T, class F, class A>
  ublas::vector<T, A> prod(ublas::matrix<T, F, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
  {
	ublas::vector<T, A> result(a.size1());
	prod(a, b, result, queue);
	return result;
  }


  //Vector-Matrix multiplication

  /**This function computes the product of vector*matrix (a*b) and stores it at vector result all 3 are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a vector A of the product (A*B) that is on device
  * \param b matrix B of the product (A*B) that is on the device
  * \param result vector on device to store the product of the result of (A*B)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype 
  * \tparam L layout of the matrix (row_major or column_major)
  */
  template <class T, class F>
  void prod(ublas::vector<T, opencl::storage>& a, ublas::matrix<T, F, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue & queue)
  {
	//check all matrices are on same context
	assert((a.device() == b.device()) && (a.device() == result.device()) && (a.device() == queue.get_device()));


	//check dimension of matricx and vector (1xN) * (NxM)
	assert(a.size() == b.size1());


	result.fill(0, queue);

	cl_event event = NULL;

	clblasOrder Order = std::is_same<F, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
	int lda = Order == clblasRowMajor ? a.size() : 1;
	int ldb = Order == clblasRowMajor ? b.size2() : a.size();
	int ldc = Order == clblasRowMajor ? b.size2() : 1;



	if (std::is_same<T, float>::value)
	  //Call clBLAS extended function. Perform gemm for float
	  clblasSgemm(Order, clblasNoTrans, clblasNoTrans,
		1, b.size2(), a.size(),
		1, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, 1,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);


	else if (std::is_same<T, double>::value)
	  //Call clBLAS extended function. Perform gemm for double
	  clblasDgemm(Order, clblasNoTrans, clblasNoTrans,
		1, b.size2(), a.size(),
		1, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, 1,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);

	else if (std::is_same<T, std::complex<float>>::value)
	  //Call clBLAS extended function. Perform gemm for complext float
	  clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
		1, b.size2(), a.size(),
		ONE_FLOAT_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_FLOAT_COMPLEX,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);

	else if (std::is_same<T, std::complex<double>>::value)
	  //Call clBLAS extended function. Perform gemm for complex double
	  clblasZgemm(Order, clblasNoTrans, clblasNoTrans,
		1, b.size2(), a.size(),
		ONE_DOUBLE_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_DOUBLE_COMPLEX,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);



	//Wait for calculations to be finished.
	clWaitForEvents(1, &event);



  }





  /**This function computes the product of vector*matrix not on device (a*b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to vector result
  *
  * \param a vector A of the product (A*B) that is not on device
  * \param b matrix B of the product (A*B) that is not on the device
  * \param result matrix on device to store the product of the result of (A*B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype
  * \tparam L layout of the matrix (row_major or column_major)
  * \tparam A storage type that has the data
  */
  template <class T, class F, class A>
  void prod(ublas::vector<T, A>& a, ublas::matrix<T, F, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::vector<T, opencl::storage> aHolder(a.size(), queue.get_context());
	aHolder.from_host(a, queue);

	///copy the data from b to bHolder
	ublas::matrix<T, F, opencl::storage> bHolder(b.size1(), b.size2(), queue.get_context());
	bHolder.from_host(b, queue);

	ublas::vector<T, opencl::storage> resultHolder(b.size2(), queue.get_context());

	prod(aHolder, bHolder, resultHolder, queue); //call the prod function that multiplies a function already on gpu

	resultHolder.to_host(result, queue);


  }


  /**This function computes the product of vector*matrix not on device (a*b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a vector A of the product (A*B) that is not on device (it's on the host)
  * \param b matrix B of the product (A*B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
   * \tparam T datatype
  * \tparam L layout of the matrix (row_major or column_major)
  * \tparam A storage type that has the data
  */

  template <class T, class F, class A>
  ublas::vector<T, A> prod(ublas::vector<T, A>& a, ublas::matrix<T, F, A>& b, compute::command_queue &queue)
  {
	ublas::vector<T, A> result(b.size2());
	prod(a, b, result, queue);
	return result;
  }



}//opencl

}//ublas
}//numeric
}//boost


#endif 
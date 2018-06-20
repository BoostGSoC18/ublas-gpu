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
template <class T, class L>
  void prod(ublas::matrix<T, L, opencl::storage>& a, ublas::matrix<T, L, opencl::storage>& b, ublas::matrix<T, L, opencl::storage>& result , compute::command_queue & queue)
{
  //check all matrices are on same context
  assert(  (a.device() == b.device()) && (a.device() == result.device()) && (a.device()== queue.get_device()) );

  //check dimension of matrices (MxN) * (NxK)
  assert(a.size2() == b.size1());

  result.fill(0, queue);

  cl_event event = NULL;

  clblasOrder Order = std::is_same<L, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
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
template <class T, class L, class A>
  void prod(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, ublas::matrix<T, L, A>& result, compute::command_queue &queue)
{

  ///copy the data from a to aHolder
  ublas::matrix<T, L, opencl::storage> aHolder(a.size1(), a.size2(), queue.get_context());
  aHolder.from_host(a,queue);

  ///copy the data from b to bHolder
  ublas::matrix<T, L, opencl::storage> bHolder(b.size1(), b.size2() ,queue.get_context());
  bHolder.from_host(b,queue);

  ublas::matrix<T, L, opencl::storage> resultHolder(a.size1(), b.size2(), queue.get_context());

  prod(aHolder, bHolder, resultHolder, queue); //call the prod function that multiplies

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

template <class T, class L, class A>
  ublas::matrix<T, L, A> prod(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, compute::command_queue &queue)
{
  ublas::matrix<T, L, A> result(a.size1(), b.size2());
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
  template <class T, class L>
  void prod(ublas::matrix<T, L, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue & queue)
  {
	//check all matrices are on same context
	assert((a.device() == b.device()) && (a.device() == result.device()) && (a.device() == queue.get_device()));


	//check dimension of matricx and vector (MxN) * (Nx1)
	assert(a.size2() == b.size());


	result.fill(0, queue);

	cl_event event = NULL;

	clblasOrder Order = std::is_same<L, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
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
  template <class T, class L, class A>
  void prod(ublas::matrix<T, L, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::matrix<T, L, opencl::storage> aHolder(a.size1(), a.size2(), queue.get_context());
	aHolder.from_host(a, queue);

	///copy the data from b to bHolder
	ublas::vector<T, opencl::storage> bHolder(b.size(), queue.get_context());
	bHolder.from_host(b, queue);

	ublas::vector<T, opencl::storage> resultHolder(a.size1(), queue.get_context());

	prod(aHolder, bHolder, resultHolder, queue); //call the prod function that multiplies

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

  template <class T, class L, class A>
  ublas::vector<T, A> prod(ublas::matrix<T, L, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
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
  template <class T, class L>
  void prod(ublas::vector<T, opencl::storage>& a, ublas::matrix<T, L, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue & queue)
  {
	//check all matrices are on same context
	assert((a.device() == b.device()) && (a.device() == result.device()) && (a.device() == queue.get_device()));


	//check dimension of matrix and vector (1xN) * (NxM)
	assert(a.size() == b.size1());


	result.fill(0, queue);

	cl_event event = NULL;

	clblasOrder Order = std::is_same<L, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
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
  template <class T, class L, class A>
  void prod(ublas::vector<T, A>& a, ublas::matrix<T, L, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::vector<T, opencl::storage> aHolder(a.size(), queue.get_context());
	aHolder.from_host(a, queue);

	///copy the data from b to bHolder
	ublas::matrix<T, L, opencl::storage> bHolder(b.size1(), b.size2(), queue.get_context());
	bHolder.from_host(b, queue);

	ublas::vector<T, opencl::storage> resultHolder(b.size2(), queue.get_context());

	prod(aHolder, bHolder, resultHolder, queue); //call the prod function that multiplies 

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

  template <class T, class L, class A>
  ublas::vector<T, A> prod(ublas::vector<T, A>& a, ublas::matrix<T, L, A>& b, compute::command_queue &queue)
  {
	ublas::vector<T, A> result(b.size2());
	prod(a, b, result, queue);
	return result;
  }


  //Elements-wise operations
  
  //matrix-matrix addition
  




  /**This function computes an element-wise operation of 2 matrices and stores it at matrix result all 3 matrices are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a matrix A of the element-wise operation that is on device
  * \param b matrix B of the element-wise operation that is on the device
  * \param result matrix on device to store the result
  * \param fun is a boost::compute binary function that is the binary operation that gets executed
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  */
  template <class T, class L, class binary_operator>
  void element_wise(ublas::matrix<T, L, opencl::storage>& a, ublas::matrix<T, L, opencl::storage>& b, ublas::matrix<T, L, opencl::storage>& result, binary_operator fun, compute::command_queue& queue)
  {
	//check all matrices are on same context
	assert((a.device() == b.device()) && (a.device() == result.device()) && (a.device() == queue.get_device()));


	//check that dimensions of matrices are equal
	assert((a.size1() == b.size1()) && (a.size2() == b.size2()));

	compute::transform(a.begin(),
	  a.end(),
	  b.begin(),
	  result.begin(),
	  fun,
	  queue);

	queue.finish();
  }



  /**This function computes an element-wise operation of 2 matrices not on device and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to matrix result
  *
  * \param a matrix A of the element-wise operation that is not on device
  * \param b matrix B of the element-wise operation that is not on the device
  * \param result matrix on device to store the operation of the result
  * \param fun is a boost::compute binary function that is the binary operation that gets executed
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L, class A, class binary_operator>
  void element_wise(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, ublas::matrix<T, L, A>& result, binary_operator fun, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::matrix<T, L, opencl::storage> aHolder(a.size1(), a.size2(), queue.get_context());
	aHolder.from_host(a, queue);

	///copy the data from b to bHolder
	ublas::matrix<T, L, opencl::storage> bHolder(b.size1(), b.size2(), queue.get_context());
	bHolder.from_host(b, queue);

	ublas::matrix<T, L, opencl::storage> resultHolder(a.size1(), b.size2(), queue.get_context());

	element_wise(aHolder, bHolder, resultHolder, fun, queue); //call the add function that performs sumition

	resultHolder.to_host(result, queue);
  }


  /**This function computation element-wise operation of 2 matrices not on device and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a matrix A of the operation that is not on device (it's on the host)
  * \param b matrix B of the operation that is not on the device (it's on the host)
  * \param fun is a boost::compute binary function that is the binary operation that gets executed
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L, class A, class binary_operator>
  ublas::matrix<T, L, A> element_wise(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, binary_operator fun, compute::command_queue &queue)
  {
	ublas::matrix<T, L, A> result(a.size1(), b.size2());
	element_wise(a, b, result, fun, queue);
	return result;
  }



  /**This function computes an element-wise operation of 2 vectors and stores it at vector result all 3 vectors are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a vector A of the element-wise operation that is on device
  * \param b vector B of the element-wise operation that is on the device
  * \param result vector on device to store the result
  * \param fun is a boost::compute binary function that is the binary operation that gets executed
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class binary_operator>
  void element_wise(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, binary_operator fun, compute::command_queue& queue)
  {
	//check all vectors are on same device
	assert((a.device() == b.device()) && (a.device() == result.device()) && (a.device() == queue.get_device()));


	//check that dimensions of matrices are equal
	assert(a.size() == b.size());

	compute::transform(a.begin(),
	  a.end(),
	  b.begin(),
	  result.begin(),
	  fun,
	  queue);

	queue.finish();
  }


  /**This function computes an element-wise operation of 2 vectors not on device  and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to vector result
  *
  * \param a vector A of the operation that is not on device
  * \param b vector B of the operation that is not on the device
  * \param result vector on device to store the result
  * \param fun is a boost::compute binary function that is the binary operation that gets executed
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */
  template <class T, class A, class binary_operator>
  void element_wise(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, binary_operator fun, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::vector<T, opencl::storage> aHolder(a.size(), queue.get_context());
	aHolder.from_host(a, queue);

	///copy the data from b to bHolder
	ublas::vector<T, opencl::storage> bHolder(b.size(), queue.get_context());
	bHolder.from_host(b, queue);

	ublas::vector<T, opencl::storage> resultHolder(a.size(), queue.get_context());

	element_wise(aHolder, bHolder, resultHolder, fun, queue); //call the add function that performs sumition

	resultHolder.to_host(result, queue);


  }

  /**This function computes an element wise operation of 2 vectors not on device and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a vector A of the operation that is not on device (it's on the host)
  * \param b vector B of the operation that is not on the device (it's on the host)
  * \param fun is a boost::compute binary function that is the binary operation that gets executed
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */
  template <class T, class A, class binary_operator>
  ublas::vector<T, A> element_wise(ublas::vector<T, A>& a, ublas::vector<T, A>& b, binary_operator fun, compute::command_queue &queue)
  {
	ublas::vector<T, A> result(a.size());
	element_wise(a, b, result, fun, queue);
	return result;
  }












  


  /**This function computes the summition (element-wise) of 2 matrices (a+b) and stores it at matrix result all 3 matrices are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a matrix A of the summition (A+B) that is on device
  * \param b matrix B of the summition (A+B) that is on the device
  * \param result matrix on device to store the result of (A+B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  */
  template <class T, class L>
  void add(ublas::matrix<T, L, opencl::storage>& a, ublas::matrix<T, L, opencl::storage>& b, ublas::matrix<T, L, opencl::storage>& result, compute::command_queue& queue)
  {
	element_wise(a, b, result, compute::plus<T>(), queue);
  }


  /**This function computes the summition (element-wise) of 2 matrices not on device (a+b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to matrix result
  *
  * \param a matrix A of the summition (A+B) that is not on device
  * \param b matrix B of the summition (A+B) that is not on the device
  * \param result matrix on device to store the summitiom of the result of (A+B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L, class A>
  void add(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, ublas::matrix<T, L, A>& result, compute::command_queue &queue)
  {
	element_wise(a, b, result, compute::plus<T>(), queue);
  }


  /**This function computes the summition (element-wise) of 2 matrices not on device (a+b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a matrix A of the summition (A+B) that is not on device (it's on the host)
  * \param b matrix B of the summition (A+B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */

  template <class T, class L, class A>
  ublas::matrix<T, L, A> add(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::plus<T>(), queue);
  }




  //vector-vector addition 


  /**This function computes the summition (element-wise) of 2 vectors (a+b) and stores it at matrix result all 3 vectors are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a vector A of the summition (A+B) that is on device
  * \param b vector B of the summition (A+B) that is on the device
  * \param result vector on device to store the result of (A+B)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  */
  template <class T>
  void add(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
  {
	element_wise(a, b, result, compute::plus<T>(), queue);
  }


  /**This function computes the summition (element-wise) of 2 vectors not on device (a+b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to vector result
  *
  * \param a vector A of the summition (A+B) that is not on device
  * \param b vector B of the summition (A+B) that is not on the device
  * \param result vector on device to store the summition of the result of (A+B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */
  template <class T, class A>
  void add(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {
	element_wise(a, b, result, compute::plus<T>(), queue);
  }


  /**This function computes the summition (element-wise) of 2 vectors not on device (a+b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a vector A of the summition (A+B) that is not on device (it's on the host)
  * \param b vector B of the summition (A+B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */
  template <class T, class A>
  ublas::vector<T, A> add(ublas::vector<T, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::plus<T>(), queue);
  }




  //matrix-matrix subtraction



  /**This function computes the subtraction (element-wise) of 2 matrices (a-b) and stores it at matrix result all 3 matrices are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a matrix A of the subtraction (A-B) that is on device
  * \param b matrix B of the subtraction (A-B) that is on the device
  * \param result matrix on device to store the result of (A-B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  */
  template <class T, class L>
  void sub(ublas::matrix<T, L, opencl::storage>& a, ublas::matrix<T, L, opencl::storage>& b, ublas::matrix<T, L, opencl::storage>& result, compute::command_queue& queue)
  {
	element_wise(a, b, compute::minus<T>(), result, queue);
  }


  /**This function computes the subtraction (element-wise) of 2 matrices not on device (a-b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to matrix result
  *
  * \param a matrix A of the subtraction (A-B) that is not on device
  * \param b matrix B of the subtraction (A-B) that is not on the device
  * \param result matrix on device to store the subtraction of the result of (A-B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L, class A>
  void sub(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, ublas::matrix<T, L, A>& result, compute::command_queue &queue)
  {
	element_wise(a, b, result, compute::minus<T>(), queue);
  }


  /**This function computes the subtraction (element-wise) of 2 matrices not on device (a-b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a matrix A of the subtraction (A-B) that is not on device (it's on the host)
  * \param b matrix B of the subtraction (A-B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */

  template <class T, class L, class A>
  ublas::matrix<T, L, A> sub(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::minus<T>(), queue);
  }




  //vector-vector subtraction 


  /**This function computes the subtraction (element-wise) of 2 vectors (a-b) and stores it at matrix result all 3 vectors are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a vector A of the subtraction (A-B) that is on device
  * \param b vector B of the subtraction (A-B) that is on the device
  * \param result vector on device to store the result of (A-B)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  */
  template <class T>
  void sub(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
  {
	element_wise(a, b, result, compute::minus<T>(), queue);
  }


  /**This function computes the subtraction (element-wise) of 2 vectors not on device (a-b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to vector result
  *
  * \param a vector A of the subtraction (A-B) that is not on device
  * \param b vector B of the subtraction (A-B) that is not on the device
  * \param result vector on device to store the subtraction of the result of (A-B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */
  template <class T, class A>
  void sub(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {
	element_wise(a, b, result, compute::minus<T>(), queue);
  }


  /**This function computes the subtraction (element-wise) of 2 vectors not on device (a-b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a vector A of the subtraction (A-B) that is not on device (it's on the host)
  * \param b vector B of the subtraction (A-B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */

  template <class T, class A>
  ublas::vector<T, A> sub(ublas::vector<T, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::minus<T>(), queue);
  }






  //matrix-matrix multiplication (element-wise)



  /**This function computes the multiplication (element-wise) of 2 matrices (a*b) and stores it at matrix result all 3 matrices are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a matrix A of the  multiplication (element-wise) (A*B) that is on device
  * \param b matrix B of the  multiplication (element-wise) (A*B) that is on the device
  * \param result matrix on device to store the result of (A*B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  */
  template <class T, class L>
  void mul(ublas::matrix<T, L, opencl::storage>& a, ublas::matrix<T, L, opencl::storage>& b, ublas::matrix<T, L, opencl::storage>& result, compute::command_queue& queue)
  {
	element_wise(a, b, result, compute::multiplies<T>(), queue);
  }


  /**This function computes the multiplication (element-wise) of 2 matrices not on device (a*b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to matrix result
  *
  * \param a matrix A of the  multiplication (element-wise) (A*B) that is not on device
  * \param b matrix B of the  multiplication (element-wise) (A*B) that is not on the device
  * \param result matrix on device to store the  multiplication (element-wise) of the result of (A*B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L, class A>
  void mul(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, ublas::matrix<T, L, A>& result, compute::command_queue &queue)
  {
	element_wise(a, b, result, compute::multiplies<T>(), queue);
  }


  /**This function computes the  multiplication (element-wise) of 2 matrices not on device (a*b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a matrix A of the  multiplication (element-wise) (A*B) that is not on device (it's on the host)
  * \param b matrix B of the  multiplication (element-wise) (A*B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L layout of the matrices (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */

  template <class T, class L, class A>
  ublas::matrix<T, L, A> mul(ublas::matrix<T, L, A>& a, ublas::matrix<T, L, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::multiplies<T>(), queue);
  }



  //vector-vector  multiplication (element-wise) 


  /**This function computes the  multiplication (element-wise) of 2 vectors (a*b) and stores it at matrix result all 3 vectors are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a vector A of the  multiplication (element-wise) (A*B) that is on device
  * \param b vector B of the  multiplication (element-wise) (A*B) that is on the device
  * \param result vector on device to store the result of (A*B)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  */
  template <class T>
  void mul(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
  {
	element_wise(a, b, result, compute::multiplies<T>(), queue);
  }


  /**This function computes the  multiplication (element-wise) of 2 vectors not on device (a*b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to vector result
  *
  * \param a vector A of the  multiplication (element-wise) (A*B) that is not on device
  * \param b vector B of the  multiplication (element-wise) (A*B) that is not on the device
  * \param result vector on device to store the  multiplication (element-wise) of the result of (A*B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */
  template <class T, class A>
  void mul(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {
	element_wise(a, b, result, compute::multiplies<T>(), queue);
  }


  /**This function computes the  multiplication (element-wise) of 2 vectors not on device (a*b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a vector A of the  multiplication (element-wise) (A*B) that is not on device (it's on the host)
  * \param b vector B of the  multiplication (element-wise) (A*B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */

  template <class T, class A>
  ublas::vector<T, A> mul(ublas::vector<T, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::multiplies<T>(), queue);
  }








}//opencl

}//ublas
}//numeric
}//boost


#endif 
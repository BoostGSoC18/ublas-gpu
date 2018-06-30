#ifndef OPENCL_OPERATIONS
#define OPENCL_OPERATIONS

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/compute/buffer.hpp>
#include <type_traits>

/// Include the clBLAS header. It includes the appropriate OpenCL headers
#include <clBLAS.h>

namespace boost {
namespace numeric {
namespace ublas {


namespace opencl
{
namespace compute = boost::compute;
namespace ublas = boost::numeric::ublas;
namespace compute_lambda = boost::compute::lambda;


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
* \tparam L1 layout of the first matrix matrix (row_major or column_major)
* \tparam L2 layout of the second matrix matrix (row_major or column_major)
*/
template <class T, class L1, class L2>

typename std::enable_if<std::is_same<T, float>::value |
  std::is_same<T, double>::value |
  std::is_same<T, std::complex<float>>::value |
  std::is_same<T, std::complex<double>>::value, 
  void>::type
prod(ublas::matrix<T, L1, opencl::storage>& a, ublas::matrix<T, L2, opencl::storage>& b, ublas::matrix<T, L1, opencl::storage>& result , compute::command_queue & queue)
{

  //check all matrices are on same context
  assert(  (a.device() == b.device()) && (a.device() == result.device()) && (a.device()== queue.get_device()) );

  //check dimension of matrices (MxN) * (NxK)
  assert(a.size2() == b.size1());

  result.fill(0, queue);

  ublas::matrix<T, L1, opencl::storage>* b_L1 = NULL; //to hold matrix b with layout 1 if the b has different layout 

 

  cl_event event = NULL;

  cl_mem buffer_a = (cl_mem)a.begin().get_buffer().get();
  cl_mem buffer_b = (cl_mem)b.begin().get_buffer().get();
  cl_mem buffer_result = (cl_mem)result.begin().get_buffer().get();



  if (!(std::is_same<L1, L2>::value))
  {
	b_L1 = new ublas::matrix<T, L1, opencl::storage>(b.size1(), b.size2(), queue.get_context());
	change_layout(b, (*b_L1), queue);
	buffer_b = (cl_mem)b_L1->begin().get_buffer().get();
  }



  clblasOrder Order = std::is_same<L1, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
  int lda = Order == clblasRowMajor ? a.size2() : a.size1();
  int ldb = Order == clblasRowMajor ? b.size2() : a.size2();
  int ldc = Order == clblasRowMajor ? b.size2() : a.size1();



  if (std::is_same<T, float>::value)
	//Call clBLAS extended function. Perform gemm for float
	clblasSgemm(Order, clblasNoTrans, clblasNoTrans,
	  a.size1(), b.size2(), a.size2(),
	  1, buffer_a, 0, lda,
	  buffer_b, 0, ldb, 1,
	  buffer_result, 0, ldc,
	  1, &(queue.get()), 0, NULL, &event);


  else if (std::is_same<T, double>::value)
	//Call clBLAS extended function. Perform gemm for double
	clblasDgemm(Order, clblasNoTrans, clblasNoTrans,
	  a.size1(), b.size2(), a.size2(),
	  1, buffer_a, 0, lda,
	  buffer_b, 0, ldb, 1,
	  buffer_result, 0, ldc,
	  1, &(queue.get()), 0, NULL, &event);

  else if (std::is_same<T, std::complex<float>>::value)
	//Call clBLAS extended function. Perform gemm for complext float
	clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
	  a.size1(), b.size2(), a.size2(),
	  ONE_FLOAT_COMPLEX, buffer_a, 0, lda,
	  buffer_b, 0, ldb, ONE_FLOAT_COMPLEX,
	  buffer_result, 0, ldc,
	  1, &(queue.get()), 0, NULL, &event);

  else if (std::is_same<T, std::complex<double>>::value)
	//Call clBLAS extended function. Perform gemm for complex double
	clblasZgemm(Order, clblasNoTrans, clblasNoTrans,
	  a.size1(), b.size2(), a.size2(),
	  ONE_DOUBLE_COMPLEX, buffer_a, 0, lda,
	  buffer_b, 0, ldb, ONE_DOUBLE_COMPLEX,
	  buffer_result, 0, ldc,
	  1, &(queue.get()), 0, NULL, &event);



  //Wait for calculations to be finished.
  clWaitForEvents(1, &event);

  if (b_L1 != NULL) delete b_L1;


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
* \tparam L1 layout of the first matrix matrix (row_major or column_major)
* \tparam L2 layout of the second matrix matrix (row_major or column_major)
* \tparam A storage type that has the data of the matrices
*/
template <class T, class L1, class L2, class A>
typename std::enable_if<std::is_same<T, float>::value |
  std::is_same<T, double>::value |
  std::is_same<T, std::complex<float>>::value |
  std::is_same<T, std::complex<double>>::value,
  void>::type
  prod(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, ublas::matrix<T, L1, A>& result, compute::command_queue &queue)
{

  ///copy the data from a to aHolder
  ublas::matrix<T, L1, opencl::storage> aHolder(a, queue);

  ///copy the data from b to bHolder
  ublas::matrix<T, L2, opencl::storage> bHolder(b, queue);

  ublas::matrix<T, L1, opencl::storage> resultHolder(a.size1(), b.size2(), queue.get_context());

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
* \tparam L1 layout of the first matrix matrix (row_major or column_major)
* \tparam L2 layout of the second matrix matrix (row_major or column_major)
* \tparam A storage type that has the data of the matrices
*/

template <class T, class L1, class L2, class A>
typename std::enable_if<std::is_same<T, float>::value |
  std::is_same<T, double>::value |
  std::is_same<T, std::complex<float>>::value |
  std::is_same<T, std::complex<double>>::value,
  ublas::matrix<T,L1,A>>::type
  prod(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, compute::command_queue &queue)
{
  ublas::matrix<T, L1, A> result(a.size1(), b.size2());
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
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	prod(ublas::matrix<T, L, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue & queue)
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
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	prod(ublas::matrix<T, L, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::matrix<T, L, opencl::storage> aHolder(a, queue);

	///copy the data from b to bHolder
	ublas::vector<T, opencl::storage> bHolder(b, queue);

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
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	ublas::vector<T,A>>::type
	prod(ublas::matrix<T, L, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
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
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	prod(ublas::vector<T, opencl::storage>& a, ublas::matrix<T, L, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue & queue)
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
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	prod(ublas::vector<T, A>& a, ublas::matrix<T, L, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::vector<T, opencl::storage> aHolder(a, queue);

	///copy the data from b to bHolder
	ublas::matrix<T, L, opencl::storage> bHolder(b, queue);

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
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	ublas::vector<T,A>>::type
	prod(ublas::vector<T, A>& a, ublas::matrix<T, L, A>& b, compute::command_queue &queue)
  {
	ublas::vector<T, A> result(b.size2());
	prod(a, b, result, queue);
	return result;
  }

  //inner product

  /** This function computes the inner product of two vector that are already on opencl device
  *
  * \param a first vector of inner product
  * \param b second vector of inner product
  * \param init initial value to start accumlating on
  * \param queue the command queue which it's device has the 2 vectors and which will execute the operation
  *
  *
  * \tparam T datatype
  * \tparam A storage type that has the data
  */
  template<class T>
  typename std::enable_if <(std::is_fundamental<T> ::value),
	T>::type
	inner_prod(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, T init, compute::command_queue& queue)
  {
	//check that both vectors are on the same device
	assert((a.device() == b.device()) && (a.device() == queue.get_device()));

	//check both vectors are the same size
	assert(a.size() == b.size());

	return compute::inner_product(a.begin(), a.end(), b.begin(), init, queue);
  }


  /** This function computes the inner product of two vector that are on host
  *
  * \param a first vector of inner product
  * \param b second vector of inner product
  * \param init initial value to start accumlating on
  * \param queue the command queue which it's device has the 2 vectors and which will execute the operation
  *
  *
  * \tparam T datatype
  * \tparam A storage type that has the data
  */
  template<class T, class A>
  typename std::enable_if <(std::is_fundamental<T> ::value),
	T>::type
	inner_prod(ublas::vector<T, A>& a, ublas::vector<T, A>& b, T init, compute::command_queue& queue)
  {
	ublas::vector<T, opencl::storage> aHolder(a, queue);

	ublas::vector<T, opencl::storage> bHolder(b, queue);

	return inner_prod(aHolder, bHolder, init, queue);
  }



  //Outer product

  /**This function computes the outer product of two vectors (a x b) and stores it at vector result all 3 are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a vector A of the outer product (A x B) that is on device
  * \param b vector B of the outer product (A x B) that is on the device
  * \param result vector on device to store the result of the outer product
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype
  * \tparam L layout of the matrix (row_major or column_major)
  */
  template <class T, class L>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	outer_prod(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::matrix<T, L, opencl::storage>& result, compute::command_queue & queue)
  {
	//check all vectors are on same context
	assert((a.device() == b.device()) && (a.device() == result.device()) && (a.device() == queue.get_device()));



	result.fill(0, queue);

	cl_event event = NULL;


	clblasOrder Order = std::is_same<L, ublas::basic_row_major<> >::value ? clblasRowMajor : clblasColumnMajor;
	int lda = Order == clblasRowMajor ? 1 : a.size();
	int ldb = Order == clblasRowMajor ? b.size() : 1;
	int ldc = Order == clblasRowMajor ? b.size() : a.size();





	if (std::is_same<T, float>::value)
	  //Call clBLAS extended function. Perform gemm for float
	  clblasSgemm(Order, clblasNoTrans, clblasNoTrans,
		a.size(), b.size(), 1,
		1, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, 1,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);


	else if (std::is_same<T, double>::value)
	  //Call clBLAS extended function. Perform gemm for double
	  clblasDgemm(Order, clblasNoTrans, clblasNoTrans,
		a.size(), b.size(), 1,
		1, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, 1,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);

	else if (std::is_same<T, std::complex<float>>::value)
	  //Call clBLAS extended function. Perform gemm for complext float
	  clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
		a.size(), b.size(), 1,
		ONE_FLOAT_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_FLOAT_COMPLEX,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);

	else if (std::is_same<T, std::complex<double>>::value)
	  //Call clBLAS extended function. Perform gemm for complex double
	  clblasZgemm(Order, clblasNoTrans, clblasNoTrans,
		a.size(), b.size(), 1,
		ONE_DOUBLE_COMPLEX, (cl_mem)a.begin().get_buffer().get(), 0, lda,
		(cl_mem)b.begin().get_buffer().get(), 0, ldb, ONE_DOUBLE_COMPLEX,
		(cl_mem)result.begin().get_buffer().get(), 0, ldc,
		1, &(queue.get()), 0, NULL, &event);



	//Wait for calculations to be finished.
	clWaitForEvents(1, &event);



  }



  /**This function computes the outer product of two vectors on host (a x b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to vector result
  *
  * \param a vector A of the product (A x B) that is not on device
  * \param b vector B of the product (A x B) that is not on the device
  * \param result matrix on device to store the result of the outer product of (A x B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype
  * \tparam L layout of the matrix (row_major or column_major)
  * \tparam A storage type that has the data
  */
  template <class T, class L, class A>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	outer_prod(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::matrix<T, L, A>& result, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::vector<T, opencl::storage> aHolder(a, queue);

	///copy the data from b to bHolder
	ublas::vector<T, opencl::storage> bHolder(b, queue);

	ublas::matrix<T, L, opencl::storage> resultHolder(a.size(), b.size(), queue.get_context());

	outer_prod(aHolder, bHolder, resultHolder, queue); //call the prod function that multiplies 

	resultHolder.to_host(result, queue);


  }


  /**This function computes the outer product of two vectors not on device (a x b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a vector A of the product (A x B) that is not on device (it's on the host)
  * \param b vector B of the product (A x B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype
  * \tparam L layout of the matrix (row_major or column_major)
  * \tparam A storage type that has the data
  */

  template <class T,class L = ublas::basic_row_major<>, class A>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	ublas::matrix<T, L, A>>::type
	outer_prod(ublas::vector<T, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue )
  {
	ublas::matrix<T, L, A> result(a.size(), b.size());
	outer_prod(a, b, result, queue);
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  */
  template <class T, class L1, class L2, class binary_operator>
  void element_wise(ublas::matrix<T, L1, opencl::storage>& a, ublas::matrix<T, L2, opencl::storage>& b, ublas::matrix<T, L1, opencl::storage>& result, binary_operator fun, compute::command_queue& queue)
  {
	//check all matrices are on same context
	assert((a.device() == b.device()) && (a.device() == result.device()) && (a.device() == queue.get_device()));


	//check that dimensions of matrices are equal
	assert((a.size1() == b.size1()) && (a.size2() == b.size2()));


	bool flag_different_layout = false;
	ublas::matrix<T, L1, opencl::storage>* b_L1 = NULL;

	if (!(std::is_same<L1, L2>::value))
	{
	  b_L1 = new ublas::matrix<T, L1, opencl::storage>(b.size1(), b.size2(), queue.get_context());
	  change_layout(b, (*b_L1), queue);
	  flag_different_layout = true;
	}


	if (flag_different_layout == false)
	{
	  compute::transform(a.begin(),
		a.end(),
		b.begin(),
		result.begin(),
		fun,
		queue);
	}

	else
	{
	  compute::transform(a.begin(),
		a.end(),
		b_L1->begin(),
		result.begin(),
		fun,
		queue);

	}

	queue.finish();

	if (flag_different_layout)
	{
	  delete b_L1;
	}
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L1, class L2, class A, class binary_operator>
  void element_wise(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, ublas::matrix<T, L1, A>& result, binary_operator fun, compute::command_queue &queue)
  {

	///copy the data from a to aHolder
	ublas::matrix<T, L1, opencl::storage> aHolder(a, queue);

	///copy the data from b to bHolder
	ublas::matrix<T, L2, opencl::storage> bHolder(b, queue);

	ublas::matrix<T, L1, opencl::storage> resultHolder(a.size1(), b.size2(), queue.get_context());

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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L1, class L2, class A, class binary_operator>
  ublas::matrix<T, L1, A> element_wise(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, binary_operator fun, compute::command_queue &queue)
  {
	ublas::matrix<T, L1, A> result(a.size1(), b.size2());
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  */
  template <class T, class L1, class L2>
  void element_add(ublas::matrix<T, L1, opencl::storage>& a, ublas::matrix<T, L2, opencl::storage>& b, ublas::matrix<T, L1, opencl::storage>& result, compute::command_queue& queue)
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L1, class L2, class A>
  void element_add(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, ublas::matrix<T, L1, A>& result, compute::command_queue &queue)
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */

  template <class T, class L1, class L2, class A>
  ublas::matrix<T, L1, A> element_add(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, compute::command_queue &queue)
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
  void element_add(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
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
  void element_add(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
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
  ublas::vector<T, A> element_add(ublas::vector<T, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  */
  template <class T, class L1, class L2>
  void element_sub(ublas::matrix<T, L1, opencl::storage>& a, ublas::matrix<T, L2, opencl::storage>& b, ublas::matrix<T, L1, opencl::storage>& result, compute::command_queue& queue)
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L1, class L2, class A>
  void element_sub(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, ublas::matrix<T, L1, A>& result, compute::command_queue &queue)
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */

  template <class T, class L1, class L2, class A>
  ublas::matrix<T, L1, A> element_sub(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, compute::command_queue &queue)
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
  void element_sub(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
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
  void element_sub(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
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
  ublas::vector<T, A> element_sub(ublas::vector<T, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  */
  template <class T, class L1, class L2>
  void element_prod(ublas::matrix<T, L1, opencl::storage>& a, ublas::matrix<T, L2, opencl::storage>& b, ublas::matrix<T, L1, opencl::storage>& result, compute::command_queue& queue)
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L1, class L2, class A>
  void element_prod(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, ublas::matrix<T, L1, A>& result, compute::command_queue &queue)
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
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */

  template <class T, class L1, class L2, class A>
  ublas::matrix<T, L1, A> element_prod(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, compute::command_queue &queue)
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
  void element_prod(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
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
  void element_prod(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
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
  ublas::vector<T, A> element_prod(ublas::vector<T, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::multiplies<T>(), queue);
  }



  //matrix-matrix division (element-wise)



  /**This function computes the division (element-wise) of 2 matrices (a/b) and stores it at matrix result all 3 matrices are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a matrix A of the  division (element-wise) (A/B) that is on device
  * \param b matrix B of the  division (element-wise) (A/B) that is on the device
  * \param result matrix on device to store the result of (A/B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  */
  template <class T, class L1, class L2>
  void element_div(ublas::matrix<T, L1, opencl::storage>& a, ublas::matrix<T, L2, opencl::storage>& b, ublas::matrix<T, L1, opencl::storage>& result, compute::command_queue& queue)
  {
	element_wise(a, b, result, compute::divides<T>(), queue);
  }


  /**This function computes the division (element-wise) of 2 matrices not on device (a/b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to matrix result
  *
  * \param a matrix A of the  division (element-wise) (A/B) that is not on device
  * \param b matrix B of the  division (element-wise) (A/B) that is not on the device
  * \param result matrix on device to store the  division (element-wise) of the result of (A*B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */
  template <class T, class L1, class L2, class A>
  void element_div(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, ublas::matrix<T, L1, A>& result, compute::command_queue &queue)
  {
	element_wise(a, b, result, compute::divides<T>(), queue);
  }


  /**This function computes the  division (element-wise) of 2 matrices not on device (a/b) and stores it at matrix result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a matrix A of the  division (element-wise) (A/B) that is not on device (it's on the host)
  * \param b matrix B of the  division (element-wise) (A/B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the matrices
  * \tparam L1 layout of the first matrix matrix (row_major or column_major)
  * \tparam L2 layout of the second matrix matrix (row_major or column_major)
  * \tparam A storage type that has the data of the matrices
  */

  template <class T, class L1, class L2, class A>
  ublas::matrix<T, L1, A> element_div(ublas::matrix<T, L1, A>& a, ublas::matrix<T, L2, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::divides<T>(), queue);
  }



  //vector-vector  division (element-wise) 


  /**This function computes the  division (element-wise) of 2 vectors (a/b) and stores it at matrix result all 3 vectors are on device
  *
  * a and b are originally on device (on the same device) and the result is left on the same device.
  *
  * \param a vector A of the  division (element-wise) (A/B) that is on device
  * \param b vector B of the  division (element-wise) (A/B) that is on the device
  * \param result vector on device to store the result of (A/B)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  * \tparam T datatype of the vectors
  */
  template <class T>
  void element_div(ublas::vector<T, opencl::storage>& a, ublas::vector<T, opencl::storage>& b, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
  {
	element_wise(a, b, result, compute::divides<T>(), queue);
  }


  /**This function computes the  division (element-wise) of 2 vectors not on device (a/b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied to vector result
  *
  * \param a vector A of the  division (element-wise) (A/B) that is not on device
  * \param b vector B of the  division (element-wise) (A/B) that is not on the device
  * \param result vector on device to store the  multiplication (element-wise) of the result of (A/B)
  * \param queue has the queue of the device which has the result matrix and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */
  template <class T, class A>
  void element_div(ublas::vector<T, A>& a, ublas::vector<T, A>& b, ublas::vector<T, A>& result, compute::command_queue &queue)
  {
	element_wise(a, b, result, compute::divides<T>(), queue);
  }


  /**This function computes the  division (element-wise) of 2 vectors not on device (a/b) and stores it at vector result which is also not on device
  *
  * a and b are originally not on device so they are copied to device and the evice does computatons on them and the result is copied from device and returned
  *
  * \param a vector A of the  division (element-wise) (A/B) that is not on device (it's on the host)
  * \param b vector B of the  division (element-wise) (A/B) that is not on the device (it's on the host)
  * \param queue has the queue of the device which has the result vector and which will do the computation
  *
  * \tparam T datatype of the vectors
  * \tparam A storage type that has the data of the vectors
  */

  template <class T, class A>
  ublas::vector<T, A> element_div(ublas::vector<T, A>& a, ublas::vector<T, A>& b, compute::command_queue &queue)
  {
	return element_wise(a, b, compute::divides<T>(), queue);
  }



  //Element-wise operations with constants


	//Matrix - Constant Addition

  /** This function adds a constant value to a matrix that is already on an opencl device
  *
  * \param m is the matrix that the value will be added to (its data exists on opencl device)
  * \param value is the constant value that will be added
  * \param result is the matrix that will hold the result of operation
  * \param queue is the command queue that its device will execute the computaion
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  */
  template<class T, class L>
  void element_add(ublas::matrix<T, L, opencl::storage>& m, T value, ublas::matrix<T, L, opencl::storage>& result, compute::command_queue& queue)
  {
	//check all are on same device
	assert((m.device() == result.device()) && (m.device() == queue.get_device()));

	//check dimensions
	assert((m.size1() == result.size1()) && (m.size2() == result.size2()));

	boost::compute::transform(m.begin(), m.end(), result.begin(), compute_lambda::_1 + value, queue);

	queue.finish();

  }


  /** This function adds a constant value to a matrix that is on host
  *
  * \param m is the matrix that the value will be added to (its data exists on host)
  * \param value is the constant value that will be added
  * \param result is the matrix that will hold the result of operation
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  * \tparam A is the storage type of the matrix
  */
  template<class T, class L, class A>
  void element_add(ublas::matrix<T, L, A>& m, T value, ublas::matrix<T, L, A>& result, compute::command_queue& queue)
  {
	ublas::matrix<T, L, opencl::storage> mHolder(m, queue);
	ublas::matrix<T, L, opencl::storage> resultHolder(result.size1(), result.size2(), queue.get_context());
	element_add(mHolder, value, resultHolder, queue);

	resultHolder.to_host(result, queue);
  }

  /** This function adds a constant value to a matrix that is on host and returns the result
  *
  * \param m is the matrix that the value will be added to (its data exists on host)
  * \param value is the constant value that will be added
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  * \tparam A is the storage type of the matrix
  */
  template<class T, class L, class A>
  ublas::matrix<T, L, A> element_add(ublas::matrix<T, L, A>& m, T value, compute::command_queue& queue)
  {
	ublas::matrix<T, L, A> result(m.size1(), m.size2());
	element_add(m, value, result, queue);

	return result;
  }

  //Vector - Constant addition

  /** This function adds a constant value to a vector that is already on an opencl device
  *
  * \param m is the vector that the value will be added to (its data exists on opencl device)
  * \param value is the constant value that will be added
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  */
  template<class T>
  void element_add(ublas::vector<T, opencl::storage>& v, T value, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
  {
	assert((v.device() == result.device()) && (v.device() == queue.get_device()));

	assert(v.size() == result.size());

	boost::compute::transform(v.begin(), v.end(), result.begin(), compute_lambda::_1 + value, queue);

	queue.finish();
  }

  /** This function adds a constant value to a vector that is on host
  *
  * \param m is the vector that the value will be added to (its data exists on host)
  * \param value is the constant value that will be added
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam A is the storage type of the vector
  */
  template<class T, class A>
  void element_add(ublas::vector<T, A>& v, T value, ublas::vector<T, A>& result, compute::command_queue& queue)
  {
	ublas::vector<T, opencl::storage> vHolder(v, queue);
	ublas::vector<T, opencl::storage> resultHolder(v.size(), queue.get_context());

	element_add(vHolder, value, resultHolder, queue);

	resultHolder.to_host(result, queue);
  }

  /** This function adds a constant value to a vector that is on host and the result will be returned
  *
  * \param m is the vector that the value will be added to (its data exists on host)
  * \param value is the constant value that will be added
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam A is the storage type of the vector
  */
  template<class T, class A>
  ublas::vector<T, A> element_add(ublas::vector<T, A>& v, T value, compute::command_queue& queue)
  {
	ublas::vector<T, A> result(v.size());

	element_add(v, value, result, queue);

	return result;
  }






  //Matrix - Constant subtraction

  /** This function subtracts a constant value to a matrix that is already on an opencl device
  *
  * \param m is the matrix that the value will be subtracted to (its data exists on opencl device)
  * \param value is the constant value that will be subtracted
  * \param result is the matrix that will hold the result of operation
  * \param queue is the command queue that its device will execute the computaion
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  */
  template<class T, class L>
  void element_sub(ublas::matrix<T, L, opencl::storage>& m, T value, ublas::matrix<T, L, opencl::storage>& result, compute::command_queue& queue)
  {
	//check all are on same device
	assert((m.device() == result.device()) && (m.device() == queue.get_device()));

	//check dimensions
	assert((m.size1() == result.size1()) && (m.size2() == result.size2()));

	boost::compute::transform(m.begin(), m.end(), result.begin(), compute_lambda::_1 - value, queue);

	queue.finish();

  }


  /** This function subtracts a constant value to a matrix that is on host
  *
  * \param m is the matrix that the value will be subtracted to (its data exists on host)
  * \param value is the constant value that will be subtracted
  * \param result is the matrix that will hold the result of operation
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  * \tparam A is the storage type of the matrix
  */
  template<class T, class L, class A>
  void element_sub(ublas::matrix<T, L, A>& m, T value, ublas::matrix<T, L, A>& result, compute::command_queue& queue)
  {
	ublas::matrix<T, L, opencl::storage> mHolder(m, queue);
	ublas::matrix<T, L, opencl::storage> resultHolder(result.size1(), result.size2(), queue.get_context());
	element_sub(mHolder, value, resultHolder, queue);

	resultHolder.to_host(result, queue);
  }

  /** This function subtracts a constant value to a matrix that is on host and returns the result
  *
  * \param m is the matrix that the value will be subtracted to (its data exists on host)
  * \param value is the constant value that will be subtracted
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  * \tparam A is the storage type of the matrix
  */
  template<class T, class L, class A>
  ublas::matrix<T, L, A> element_sub(ublas::matrix<T, L, A>& m, T value, compute::command_queue& queue)
  {
	ublas::matrix<T, L, A> result(m.size1(), m.size2());
	element_sub(m, value, result, queue);

	return result;
  }

  //Vector - Constant subtraction

  /** This function subtracts a constant value to a vector that is already on an opencl device
  *
  * \param m is the vector that the value will be subtracted to (its data exists on opencl device)
  * \param value is the constant value that will be subtracted
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  */
  template<class T>
  void element_sub(ublas::vector<T, opencl::storage>& v, T value, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
  {
	assert((v.device() == result.device()) && (v.device() == queue.get_device()));

	assert(v.size() == result.size());

	boost::compute::transform(v.begin(), v.end(), result.begin(), compute_lambda::_1 - value, queue);

	queue.finish();
  }

  /** This function subtracts a constant value to a vector that is on host
  *
  * \param m is the vector that the value will be subtracted to (its data exists on host)
  * \param value is the constant value that will be subtracted
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam A is the storage type of the vector
  */
  template<class T, class A>
  void element_sub(ublas::vector<T, A>& v, T value, ublas::vector<T, A>& result, compute::command_queue& queue)
  {
	ublas::vector<T, opencl::storage> vHolder(v, queue);
	ublas::vector<T, opencl::storage> resultHolder(v.size(), queue.get_context());

	element_sub(vHolder, value, resultHolder, queue);

	resultHolder.to_host(result, queue);
  }

  /** This function subtracts a constant value to a vector that is on host and the result will be returned
  *
  * \param m is the vector that the value will be subtracted to (its data exists on host)
  * \param value is the constant value that will be subtracted
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam A is the storage type of the vector
  */
  template<class T, class A>
  ublas::vector<T, A> element_sub(ublas::vector<T, A>& v, T value, compute::command_queue& queue)
  {
	ublas::vector<T, A> result(v.size());

	element_sub(v, value, result, queue);

	return result;
  }


 








  //Matrix - Constant Multiplication

  /** This function multiplies a constant value to a matrix that is already on an opencl device
  *
  * \param m is the matrix that the value will be multiplied to (its data exists on opencl device)
  * \param value is the constant value that will be multiplied
  * \param result is the matrix that will hold the result of operation
  * \param queue is the command queue that its device will execute the computaion
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  */
  template<class T, class L>
	void element_scale(ublas::matrix<T, L, opencl::storage>& m, T value, ublas::matrix<T, L, opencl::storage>& result, compute::command_queue& queue)
  {
	//check all are on same device
	assert((m.device() == result.device()) && (m.device() == queue.get_device()));

	//check dimensions
	assert((m.size1() == result.size1()) && (m.size2() == result.size2()));

	boost::compute::transform(m.begin(), m.end(), result.begin(), compute_lambda::_1 * value, queue);

	queue.finish();

  }

  /** This function multiplies a constant value to a matrix that is on host
  *
  * \param m is the matrix that the value will be multiplied to (its data exists on host)
  * \param value is the constant value that will be multiplied
  * \param result is the matrix that will hold the result of operation
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  * \tparam A is the storage type of the matrix
  */
  template<class T, class L, class A>
   void element_scale(ublas::matrix<T, L, A>& m, T value, ublas::matrix<T, L, A>& result, compute::command_queue& queue)
  {
	ublas::matrix<T, L, opencl::storage> mHolder(m, queue);
	ublas::matrix<T, L, opencl::storage> resultHolder(result.size1(), result.size2(), queue.get_context());
	element_scale(mHolder, value, resultHolder, queue);

	resultHolder.to_host(result, queue);
  }

  /** This function multiplies a constant value to a matrix that is on host and returns the result
  *
  * \param m is the matrix that the value will be multiplied to (its data exists on host)
  * \param value is the constant value that will be multiplied
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrix
  * \tparam A is the storage type of the matrix
  */
  template<class T, class L, class A>
	ublas::matrix<T, L, A>  element_scale(ublas::matrix<T, L, A>& m, T value, compute::command_queue& queue)
  {
	ublas::matrix<T, L, A> result(m.size1(), m.size2());
	element_scale(m, value, result, queue);

	return result;
  }

  //Vector - Constant multiplication

  /** This function multiplies a constant value to a vector that is already on an opencl device
  *
  * \param m is the vector that the value will be multiplied to (its data exists on opencl device)
  * \param value is the constant value that will be multiplied
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  */
  template<class T>
	void element_scale(ublas::vector<T, opencl::storage>& v, T value, ublas::vector<T, opencl::storage>& result, compute::command_queue& queue)
  {
	assert((v.device() == result.device()) && (v.device() == queue.get_device()));

	assert(v.size() == result.size());

	boost::compute::transform(v.begin(), v.end(), result.begin(), compute_lambda::_1 * value, queue);

	queue.finish();
  }

  /** This function multiplies a constant value to a vector that is on host
  *
  * \param m is the vector that the value will be multiplied to (its data exists on host)
  * \param value is the constant value that will be multiplied
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam A is the storage type of the vector
  */
  template<class T, class A>
	void element_scale(ublas::vector<T, A>& v, T value, ublas::vector<T, A>& result, compute::command_queue& queue)
  {
	ublas::vector<T, opencl::storage> vHolder(v, queue);
	ublas::vector<T, opencl::storage> resultHolder(v.size(), queue.get_context());

	element_scale(vHolder, value, resultHolder, queue);

	resultHolder.to_host(result, queue);
  }

  /** This function multiplies a constant value to a vector that is on host and the result will be returned
  *
  * \param m is the vector that the value will be multiplied to (its data exists on host)
  * \param value is the constant value that will be multiplied
  * \param queue is the command queue that its device will execute the computation
  *
  * \tparam T is the data type
  * \tparam A is the storage type of the vector
  */
  template<class T, class A>
	ublas::vector<T,A> element_scale(ublas::vector<T, A>& v, T value, compute::command_queue& queue)
  {
	ublas::vector<T, A> result(v.size());

	element_scale(v, value, result, queue);

	return result;
  }





  //Transpose


  //Kernel for transposition of various data types
#define OPENCL_TRANSPOSITION_KERNEL(DATA_TYPE)	 "__kernel void transpose(__global "  #DATA_TYPE "* in, __global " #DATA_TYPE "* result, unsigned int width, unsigned int height) \n"\
												 " { \n"\
												 "unsigned int column_index = get_global_id(0); \n"\
												 "unsigned int row_index = get_global_id(1); \n"\
												 "if (column_index < width && row_index < height) \n"\
												  "{ \n"\
												  "unsigned int index_in = column_index + width * row_index; \n"\
												  "unsigned int index_result = row_index + height * column_index; \n"\
												  "result[index_result] = in[index_in]; \n"\
												  "} \n"\
												  "} \n"


  /**This function computes the transposition of a matrix on an opencl device
  * \param m is the input matrix that will be transposed (it's already on an opencl device)
  * \param result is te matrix that will hold the result of the transposition
  * \param queue is the command queue that its device will do the computation and will have the result
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrices
  */
  template<class T, class L>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	trans(ublas::matrix<T, L, opencl::storage>& m, ublas::matrix<T, L, opencl::storage>& result, compute::command_queue& queue)
  {

	//check the right dimensions
	assert((m.size1() == result.size2()) && (m.size2() == result.size1()));

	//assert all matrices are on the same device
	assert((m.device() == result.device()) && (m.device() == queue.get_device()));

	const char* kernel;


	//decide the precision's kernel
	if (std::is_same<T, float>::value)

	  kernel = OPENCL_TRANSPOSITION_KERNEL(float);

	else if (std::is_same<T, double>::value)

	  kernel = OPENCL_TRANSPOSITION_KERNEL(double);

	else if (std::is_same<T, std::complex<float>>::value)

	  kernel = OPENCL_TRANSPOSITION_KERNEL(float2);

	else if (std::is_same<T, std::complex<double>>::value)

	  kernel = OPENCL_TRANSPOSITION_KERNEL(double2);


	size_t len = strlen(kernel);
	cl_int err;

	cl_context c_context = queue.get_context().get();
	cl_program program = clCreateProgramWithSource(c_context, 1, &kernel, &len, &err);
	clBuildProgram(program, 1, &queue.get_device().get(), NULL, NULL, NULL);




	cl_kernel c_kernel = clCreateKernel(program, "transpose", &err);

	int width = std::is_same < L, ublas::basic_row_major<>>::value ? m.size2() : m.size1();
	int height = std::is_same < L, ublas::basic_row_major<>>::value ? m.size1() : m.size2();

	size_t global_size[2] = { width , height };
	clSetKernelArg(c_kernel, 0, sizeof(T*), &m.begin().get_buffer().get());
	clSetKernelArg(c_kernel, 1, sizeof(T*), &result.begin().get_buffer().get());
	clSetKernelArg(c_kernel, 2, sizeof(unsigned int), &width);
	clSetKernelArg(c_kernel, 3, sizeof(unsigned int), &height);


	cl_command_queue c_queue = queue.get();

	cl_event event = NULL;
	clEnqueueNDRangeKernel(c_queue, c_kernel, 2, NULL, global_size, NULL, 0, NULL, &event);


	clWaitForEvents(1, &event);
  }


  /**This function computes the transposition of a matrix on host
  * \param m is the input matrix that will be transposed (it's on host)
  * \param result is te matrix that will hold the result of the transposition
  * \param queue is the command queue that its device will do the computation
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrices
  */
	template<class T, class L, class A>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	trans(ublas::matrix<T, L, A>& m, ublas::matrix<T, L, A>& result, compute::command_queue& queue)
  {
	ublas::matrix<T, L, opencl::storage> mHolder(m, queue);

	ublas::matrix<T, L, opencl::storage> resultHolder(result.size1(), result.size2(), queue.get_context());

	trans(mHolder, resultHolder, queue);

	resultHolder.to_host(result, queue);
  }


  /**This function computes the transposition of a matrix on host and returns the result as a return value
  * \param m is the input matrix that will be transposed (it's on host)
  * \param result is te matrix that will hold the result of the transposition
  * \param queue is the command queue that its device will do the computation
  *
  * \tparam T is the data type
  * \tparam L is the layout of the matrices
  */
  template<class T, class L, class A>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	ublas::matrix<T, L, A>>::type
	trans(ublas::matrix<T, L, A>& m, compute::command_queue& queue)
  {
	ublas::matrix<T, L, A> result(m.size2(), m.size1());
	trans(m, result, queue);
	return result;
  }




  //Change Layout of matrix (from row-major to column-major and vise versa)

  /**This function changes the layout of the matrix (from row-major to column-major and vise versa) of a matrix on opencl device
  * \param m is the input matrix that its layout will be changed (it's already on an opencl device)
  * \param result is te matrix that will hold the result
  * \param queue is the command queue that its device will do the computation and will have the result
  *
  * \tparam T is the data type
  * \tparam L1 is the layout of the input matrix
  * \tparam L2 is the layout of the output matrix
  */
  template<class T, class L1, class L2>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	change_layout(ublas::matrix<T, L1, opencl::storage>& m, ublas::matrix<T, L2, opencl::storage>& result, compute::command_queue& queue)
  {

	//check the right dimensions
	assert((m.size1() == result.size1()) && (m.size2() == result.size2()));

	//assert all matrices are on the same device
	assert((m.device() == result.device()) && (m.device() == queue.get_device()));

	//make sure the input layout is not the requires layout
	assert(!(std::is_same<L1, L2>::value));

	const char* kernel;


	//decide the precision's kernel
	if (std::is_same<T, float>::value)

	  kernel = OPENCL_TRANSPOSITION_KERNEL(float);

	else if (std::is_same<T, double>::value)

	  kernel = OPENCL_TRANSPOSITION_KERNEL(double);

	else if (std::is_same<T, std::complex<float>>::value)

	  kernel = OPENCL_TRANSPOSITION_KERNEL(float2);

	else if (std::is_same<T, std::complex<double>>::value)

	  kernel = OPENCL_TRANSPOSITION_KERNEL(double2);


	size_t len = strlen(kernel);
	cl_int err;

	cl_context c_context = queue.get_context().get();
	cl_program program = clCreateProgramWithSource(c_context, 1, &kernel, &len, &err);
	clBuildProgram(program, 1, &queue.get_device().get(), NULL, NULL, NULL);




	cl_kernel c_kernel = clCreateKernel(program, "transpose", &err);

	int width = std::is_same < L1, ublas::basic_row_major<>>::value ? m.size2() : m.size1();
	int height = std::is_same < L1, ublas::basic_row_major<>>::value ? m.size1() : m.size2();

	size_t global_size[2] = { width , height };
	clSetKernelArg(c_kernel, 0, sizeof(T*), &m.begin().get_buffer().get());
	clSetKernelArg(c_kernel, 1, sizeof(T*), &result.begin().get_buffer().get());
	clSetKernelArg(c_kernel, 2, sizeof(unsigned int), &width);
	clSetKernelArg(c_kernel, 3, sizeof(unsigned int), &height);


	cl_command_queue c_queue = queue.get();

	cl_event event = NULL;
	clEnqueueNDRangeKernel(c_queue, c_kernel, 2, NULL, global_size, NULL, 0, NULL, &event);


	clWaitForEvents(1, &event);
  }





  /**This function changes the layout of the matrix (from row-major to column-major and vise versa) of a matrix on host
  * \param m is the input matrix that its layout will bw changed (it's on host)
  * \param result is te matrix that will hold the result
  * \param queue is the command queue that its device will do the computation
  *
  * \tparam T is the data type
  * \tparam L1 is the layout of the input matrix
  * \tparam L2 is the layout of the output matrix
  */
  template<class T, class L1, class L2, class A>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	void>::type
	change_layout(ublas::matrix<T, L1, A>& m, ublas::matrix<T, L2, A>& result, compute::command_queue& queue)
  {
	ublas::matrix<T, L1, opencl::storage> mHolder(m, queue);

	ublas::matrix<T, L2, opencl::storage> resultHolder(result.size1(), result.size2(), queue.get_context());

	change_layout(mHolder, resultHolder, queue);

	resultHolder.to_host(result, queue);
  }


  //Absolute sum of a vector

  /** This function computes absoulte sum of v elements on opencl device
  *
  * \param v the vector on opencl device which its absolute sum will be computed
  * \param queue is the command_queue that will execute the operations
  *
  * \tparam T is the data type 
  */
  template<class T>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	T>::type a_sum(ublas::vector<T, opencl::storage>& v, compute::command_queue& queue)
  {

	//temporary buffer needed by the kernel
	compute::vector<T> scratch_buffer(v.size(), queue.get_context());

	//to create a buffer to hold the absolute sum and gain access to return the value
	compute::vector<T> result_buffer(1, queue.get_context());

	cl_event event;

	if (std::is_same<T, float>::value)
	{
	  clblasSasum(v.size(),
		(cl_mem)result_buffer.begin().get_buffer().get(), //result buffer
		0, //offset in result buffer
		(cl_mem)v.begin().get_buffer().get(), //input buffer
		0, //offset in input buffer
		1, //increment in input buffer
		(cl_mem)scratch_buffer.begin().get_buffer().get(), //scratch (temp) buffer
		1, //number of command queues
		&(queue.get()), //queue
		0, // number of events waiting list
		NULL, //event waiting list
		&event); //event
	}

	else if (std::is_same<T, double>::value)
	{
	  clblasDasum(v.size(),
		(cl_mem)result_buffer.begin().get_buffer().get(), //result buffer
		0, //offset in result buffer
		(cl_mem)v.begin().get_buffer().get(), //input buffer
		0, //offset in input buffer
		1, //increment in input buffer
		(cl_mem)scratch_buffer.begin().get_buffer().get(), //scratch (temp) buffer
		1, //number of command queues
		&(queue.get()), //queue
		0, // number of events waiting list
		NULL, //event waiting list
		&event); //event
	}

	else if (std::is_same<T, std::complex<float>>::value)
	{
	  clblasScasum(v.size(),
		(cl_mem)result_buffer.begin().get_buffer().get(), //result buffer
		0, //offset in result buffer
		(cl_mem)v.begin().get_buffer().get(), //input buffer
		0, //offset in input buffer
		1, //increment in input buffer
		(cl_mem)scratch_buffer.begin().get_buffer().get(), //scratch (temp) buffer
		1, //number of command queues
		&(queue.get()), //queue
		0, // number of events waiting list
		NULL, //event waiting list
		&event); //event
	}

	else if (std::is_same<T, std::complex<double>>::value)
	{
	  clblasDzasum(v.size(),
		(cl_mem)result_buffer.begin().get_buffer().get(), //result buffer
		0, //offset in result buffer
		(cl_mem)v.begin().get_buffer().get(), //input buffer
		0, //offset in input buffer
		1, //increment in input buffer
		(cl_mem)scratch_buffer.begin().get_buffer().get(), //scratch (temp) buffer
		1, //number of command queues
		&(queue.get()), //queue
		0, // number of events waiting list
		NULL, //event waiting list
		&event); //event
	}


	//Wait for calculations to be finished.
	clWaitForEvents(1, &event);

	return result_buffer[0];


  }



  /** This function computes absolute sum of a vector on host
  *
  * \param v the vector on host which its  absolute sum will be computed
  * \param queue is the command_queue that will execute the operations
  *
  * \tparam T is the data type
  */
  template<class T, class A>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	T>::type a_sum(ublas::vector<T, A>& v, compute::command_queue& queue)
  {
	ublas::vector<T, opencl::storage> vHolder(v, queue);

	return a_sum(vHolder, queue);
  }



  //Eucledian norm of a vector 

  /** This function computes ||v||2 on opencl device
  *
  * \param v the vector on opencl device which its norm_2 will be computed
  * \param queue is the command_queue that will execute the operations
  *
  * \tparam T is the data type
  */
  template<class T>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	T>::type norm_2(ublas::vector<T, opencl::storage>& v, compute::command_queue& queue)
  {

	//temporary buffer needed by the kernel
	compute::vector<T> scratch_buffer(2*v.size(), queue.get_context());

	//to create a buffer to hold the  ||v||2 and gain access to return the value
	compute::vector<T> result_buffer(1, queue.get_context());

	cl_event event;

	if (std::is_same<T, float>::value)
	{
	  clblasSnrm2(v.size(),
		(cl_mem)result_buffer.begin().get_buffer().get(), //result buffer
		0, //offset in result buffer
		(cl_mem)v.begin().get_buffer().get(), //input buffer
		0, //offset in input buffer
		1, //increment in input buffer
		(cl_mem)scratch_buffer.begin().get_buffer().get(), //scratch (temp) buffer
		1, //number of command queues
		&(queue.get()), //queue
		0, // number of events waiting list
		NULL, //event waiting list
		&event); //event
	}

	else if (std::is_same<T, double>::value)
	{
	  clblasDnrm2(v.size(),
		(cl_mem)result_buffer.begin().get_buffer().get(), //result buffer
		0, //offset in result buffer
		(cl_mem)v.begin().get_buffer().get(), //input buffer
		0, //offset in input buffer
		1, //increment in input buffer
		(cl_mem)scratch_buffer.begin().get_buffer().get(), //scratch (temp) buffer
		1, //number of command queues
		&(queue.get()), //queue
		0, // number of events waiting list
		NULL, //event waiting list
		&event); //event
	}

	else if (std::is_same<T, std::complex<float>>::value)
	{
	  clblasScnrm2(v.size(),
		(cl_mem)result_buffer.begin().get_buffer().get(), //result buffer
		0, //offset in result buffer
		(cl_mem)v.begin().get_buffer().get(), //input buffer
		0, //offset in input buffer
		1, //increment in input buffer
		(cl_mem)scratch_buffer.begin().get_buffer().get(), //scratch (temp) buffer
		1, //number of command queues
		&(queue.get()), //queue
		0, // number of events waiting list
		NULL, //event waiting list
		&event); //event
	}

	else if (std::is_same<T, std::complex<double>>::value)
	{
	  clblasDznrm2(v.size(),
		(cl_mem)result_buffer.begin().get_buffer().get(), //result buffer
		0, //offset in result buffer
		(cl_mem)v.begin().get_buffer().get(), //input buffer
		0, //offset in input buffer
		1, //increment in input buffer
		(cl_mem)scratch_buffer.begin().get_buffer().get(), //scratch (temp) buffer
		1, //number of command queues
		&(queue.get()), //queue
		0, // number of events waiting list
		NULL, //event waiting list
		&event); //event
	}


	//Wait for calculations to be finished.
	clWaitForEvents(1, &event);

	return result_buffer[0];


  }



  /** This function computes ||v||2 of a vector on host
  *
  * \param v the vector on host which its norm_2 will be computed
  * \param queue is the command_queue that will execute the operations
  *
  * \tparam T is the data type
  */
  template<class T, class A>
  typename std::enable_if<std::is_same<T, float>::value |
	std::is_same<T, double>::value |
	std::is_same<T, std::complex<float>>::value |
	std::is_same<T, std::complex<double>>::value,
	T>::type norm_2(ublas::vector<T, A>& v, compute::command_queue& queue)
  {
	ublas::vector<T, opencl::storage> vHolder(v, queue);

	return norm_2(vHolder, queue);
  }



}//opencl

}//ublas
}//numeric
}//boost


#endif 
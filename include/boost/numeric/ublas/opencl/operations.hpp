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




				/**
				This function computes the prodect of 2 matrices (A*B) and stores it at matrix result all 3 matrices are on device
				a and b are originally on gpu (on the same device) and the result is left on the same device
				*/
				template <class T, class F>
				BOOST_UBLAS_INLINE
					void prod(ublas::matrix<T, F, opencl::storage>& a, ublas::matrix<T, F, opencl::storage>& b, ublas::matrix<T, F, opencl::storage>& result)
				{


					///get data from device
					compute::device device = a.device().getDevice();
					compute::context context = a.device().getContext();
					compute::command_queue queue = a.device().getQueue();


					result.resize(a.size1(), b.size2());
					result.fill(0);

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
							1, (cl_mem)a.data().begin().get_buffer().get(), 0, lda,
							(cl_mem)b.data().begin().get_buffer().get(), 0, ldb, 1,
							(cl_mem)result.data().begin().get_buffer().get(), 0, ldc,
							1, &(queue.get()), 0, NULL, &event);


					else if (std::is_same<T, double>::value)
						///Call clBLAS extended function. Perform gemm for double
						err = clblasDgemm(Order, clblasNoTrans, clblasNoTrans,
							a.size1(), b.size2(), a.size2(),
							1, (cl_mem)a.data().begin().get_buffer().get(), 0, lda,
							(cl_mem)b.data().begin().get_buffer().get(), 0, ldb, 1,
							(cl_mem)result.data().begin().get_buffer().get(), 0, ldc,
							1, &(queue.get()), 0, NULL, &event);

					else if (std::is_same<T, std::complex<float>>::value)
						///Call clBLAS extended function. Perform gemm for double
						err = clblasCgemm(Order, clblasNoTrans, clblasNoTrans,
							a.size1(), b.size2(), a.size2(),
							ONE_FLOAT_COMPLEX, (cl_mem)a.data().begin().get_buffer().get(), 0, lda,
							(cl_mem)b.data().begin().get_buffer().get(), 0, ldb, ONE_FLOAT_COMPLEX,
							(cl_mem)result.data().begin().get_buffer().get(), 0, ldc,
							1, &(queue.get()), 0, NULL, &event);

					else if (std::is_same<T, std::complex<double>>::value)
						///Call clBLAS extended function. Perform gemm for double
						err = clblasZgemm(Order, clblasNoTrans, clblasNoTrans,
							a.size1(), b.size2(), a.size2(),
							ONE_DOUBLE_COMPLEX, (cl_mem)a.data().begin().get_buffer().get(), 0, lda,
							(cl_mem)b.data().begin().get_buffer().get(), 0, ldb, ONE_DOUBLE_COMPLEX,
							(cl_mem)result.data().begin().get_buffer().get(), 0, ldc,
							1, &(queue.get()), 0, NULL, &event);



					///Wait for calculations to be finished.
					clWaitForEvents(1, &event);



				}




				/**
				This function computes the prodect of 2 matrices (A*B) and stores it at matrix result and keeps result on device
				it first transfers the data of matrix a,b to the device and execute a clBlas kernel according to
				the data type of the matrices to multiply them
				*/
				template <class T, class F, class A>
				BOOST_UBLAS_INLINE
					void prod(ublas::matrix<T, F, A>& a, ublas::matrix<T, F, A>& b, ublas::matrix<T, F, opencl::storage>& result, opencl_device& device)
				{

					

				}


				/**
				This function computes the prodect of 2 matrices (A*B) and stores it at matrix result (all 3 matrices are not on device
				it first transfers the data of matrix a,b to the device and execute a clBlas kernel according to
				the data type of the matrices to multiply them
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


				/**
				This function computes the prodect of 2 matrices (A*B) that are not on gpu and returns the result
				it uses the function prod(A,B,result) and returns result
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
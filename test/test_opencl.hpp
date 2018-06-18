#ifndef TEST_OPENCL_HH
#define TEST_OPENCL_HH
#include <stdio.h>

#define ENABLE_OPENCL
#include <boost/numeric/ublas/matrix.hpp>
#include <time.h>
#include <math.h>

namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;
#define NUMBER_OF_TESTS 100
#define MAX_DIMENSION 100


template <class T, class F, int number_of_tests, int max_dimension>
class bench_prod
{

  bool compare(ublas::matrix<T, F>& a, ublas::matrix<T, F>& b)
  {
	if ((a.size1() != b.size1()) || (a.size2() != b.size2()))
	  return false;

	for (int i = 0; i<a.size1(); i++)
	  for (int j = 0; j<a.size2(); j++)
		if (a(i, j) != b(i, j))
		  return false;

	return true;

  }


  bool compare(ublas::vector<T>& a, ublas::vector<T>& b)
  {
	if (a.size() != b.size())
	  return false;

	for (int i = 0; i<a.size(); i++)
	  if (a[i] != b[i])
		return false;

	return true;

  }



  void init_matrix(ublas::matrix<T, F>& m, int max_value)
  {
	for (int i = 0; i < m.size1(); i++)
	{
	  for (int j = 0; j<m.size2(); j++)
		m(i, j) = std::rand() % max_value;

	}
  }


  void init_vector(ublas::vector<T>& v, int max_value)
  {
	for (int i = 0; i <v.size(); i++)
	{
	  v[i] = std::rand() % max_value;
	}
  }


public:


  void run()
  {
	opencl::library lib;
	int passedOperations = 0;
	// get default device and setup context
	compute::device device = compute::system::devices().at(0);
	compute::context context(device);
	compute::command_queue queue(context, device);

	std::srand(time(0));

	ublas::matrix<T, F> a;
	ublas::matrix<T, F> b;
	ublas::matrix<T, F> resultUBLAS;
	ublas::matrix<T, F> resultOPENCL;
	ublas::vector<T> va;
	ublas::vector<T> vb;
	ublas::vector<T> result_vector_ublas_mv;
	ublas::vector<T> result_vector_ublas_vm;
	ublas::vector<T> result_vector_opencl_mv;
	ublas::vector<T> result_vector_opencl_vm;



	for (int i = 0; i<number_of_tests; i++)
	{
	  int rowsA = std::rand() % max_dimension + 1;
	  int colsA = std::rand() % max_dimension + 1;
	  int colsB = std::rand() % max_dimension + 1;

	  a.resize(rowsA, colsA);
	  b.resize(colsA, colsB);
	  va.resize(colsA);
	  vb.resize(rowsA);


	  init_matrix(a, 200);
	  init_matrix(b, 200);
	  init_vector(va, 200);
	  init_vector(vb, 200);

	  //matrix_matrix
	  resultUBLAS = prod(a, b);
	  resultOPENCL = opencl::prod(a, b, queue);


	  //matrix_vector
	  result_vector_ublas_mv = ublas::prod(a, va);
	  result_vector_opencl_mv = opencl::prod(a, va, queue);


	  //vector-matrix
	  result_vector_ublas_vm = ublas::prod(vb, a);
	  result_vector_opencl_vm = opencl::prod(vb, a, queue);


	  if ((!compare(resultUBLAS, resultOPENCL)) || (!compare(result_vector_opencl_mv, result_vector_ublas_mv)) || (!compare(result_vector_opencl_vm, result_vector_ublas_vm)))
	  {
		std::cout << "Error in calculations" << std::endl;

		std::cout << "passed: " << passedOperations << std::endl;
		return;
	  }

	  passedOperations++;

	}
	std::cout << "All is well (matrix opencl prod) of " << typeid(T).name() << std::endl;



  }

};

#endif
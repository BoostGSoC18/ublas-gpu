#ifndef ELEMENT_OPENCL_HH
#define ELEMENT_OPENCL_HH
#include <stdio.h>

#define ENABLE_OPENCL
#include <boost/numeric/ublas/matrix.hpp>
#include <time.h>
#include <math.h>
#include <limits>


namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;

template <class T, class F, int number_of_tests, int max_dimension>
class bench_elementwise
{

  bool compare(ublas::matrix<T, F>& a, ublas::matrix<T, F>& b)
  {
	if ((a.size1() != b.size1()) || (a.size2() != b.size2()))
	  return false;

	for (int i = 0; i<a.size1(); i++)
	  for (int j = 0; j<a.size2(); j++)
		if (a(i, j) != b(i, j))
		{
		  std::cout << a(i, j) << " => " << b(i, j) << std::endl;
		  return false;
		}
	return true;

  }


  bool compare(ublas::vector<T>& a, ublas::vector<T>& b)
  {
	if (a.size() != b.size())
	  return false;

	for (int i = 0; i<a.size(); i++)
	  if ((a[i] != b[i]))
	  {
		std::cout << a[i] << " => " << b[i] << std::endl;
		return false;
	  }
	return true;

  }



  void init_matrix(ublas::matrix<T, F>& m, int max_value)
  {
	for (int i = 0; i < m.size1(); i++)
	{
	  for (int j = 0; j<m.size2(); j++)
		m(i, j) = (std::rand() % max_value) + 1;

	}
  }


  void init_vector(ublas::vector<T>& v, int max_value)
  {
	for (int i = 0; i <v.size(); i++)
	{
	  v[i] = std::rand() % (std::rand() % max_value) + 1;
	}
  }


public:


  void run()
  {
	opencl::library lib;
	int passedOperations = 0;
	// get default device and setup context
	compute::device device = compute::system::default_device();
	compute::context context(device);
	compute::command_queue queue(context, device);

	std::srand(time(0));

	ublas::matrix<T, F> a;
	ublas::matrix<T, F> b;

	//matrix-matrix operations of cpu
	ublas::matrix<T, F> result_m_add;
	ublas::matrix<T, F> result_m_sub;
	ublas::matrix<T, F> result_m_mul;

	//matrix-matrix operations of gpu
	ublas::matrix<T, F> result_m_add_cl;
	ublas::matrix<T, F> result_m_sub_cl;
	ublas::matrix<T, F> result_m_mul_cl;


	ublas::vector<T> va;
	ublas::vector<T> vb;

	//vector-vector operations of cpu 
	ublas::vector<T> result_v_add;
	ublas::vector<T> result_v_sub;
	ublas::vector<T> result_v_mul;

	//vector-vector operations of gpu 
	ublas::vector<T> result_v_add_cl;
	ublas::vector<T> result_v_sub_cl;
	ublas::vector<T> result_v_mul_cl;

	compare(result_m_add, result_m_add_cl);


	for (int i = 0; i<number_of_tests; i++)
	{
	  int rows = std::rand() % max_dimension + 1;
	  int cols = std::rand() % max_dimension + 1;

	  a.resize(rows, cols);
	  b.resize(rows, cols);
	  va.resize(rows);
	  vb.resize(rows);


	  init_matrix(a, 200);
	  init_matrix(b, 200);
	  init_vector(va, 200);
	  init_vector(vb, 200);

	  result_m_add = a + b;
	  result_m_add_cl = opencl::add(a, b, queue);

	  result_m_sub = a - b;
	  result_m_sub_cl = opencl::sub(a, b, queue);

	  result_m_mul = ublas::element_prod(a, b);
	  result_m_mul_cl = opencl::mul(a, b, queue);


	  result_v_add = va + vb;
	  result_v_add_cl = opencl::add(va, vb, queue);

	  result_v_sub = va - vb;
	  result_v_sub_cl = opencl::sub(va, vb, queue);

	  result_v_mul = ublas::element_prod(va, vb);
	  result_v_mul_cl = opencl::mul(va, vb, queue);






	  if ((!compare(result_m_add, result_m_add_cl)) ||
		(!compare(result_m_sub, result_m_sub_cl)) ||
		(!compare(result_m_mul, result_m_mul_cl)) ||
		(!compare(result_v_add, result_v_add_cl)) ||
		(!compare(result_v_sub, result_v_sub_cl)) ||
		(!compare(result_v_mul, result_v_mul_cl))
		)
	  {
		std::cout << "Error in calculations" << std::endl;

		std::cout << "passed: " << passedOperations << std::endl;
		return;
	  }

	  passedOperations++;

	}
	std::cout << "All is well (matrix opencl element wise operations) of " << typeid(T).name() << std::endl;



  }

};

#endif

#ifndef TEST_OPENCL_HEADER_HH
#define TEST_OPENCL_HEADER_HH
#include <stdio.h>

#define ENABLE_OPENCL
#include <boost/numeric/ublas/matrix.hpp>
#include <time.h>
#include <math.h>




namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;

template <class T, class F = ublas::basic_row_major<>>
class test_opencl
{
public:
  static bool compare(ublas::matrix<T, F>& a, ublas::matrix<T, F>& b)
  {
	if ((a.size1() != b.size1()) || (a.size2() != b.size2()))
	  return false;

	for (int i = 0; i<a.size1(); i++)
	  for (int j = 0; j<a.size2(); j++)
		if (a(i, j) != b(i, j))
		{
		  return false;
		}
	return true;

  }


  static bool compare(ublas::vector<T>& a, ublas::vector<T>& b)
  {
	if (a.size() != b.size())
	  return false;

	for (int i = 0; i<a.size(); i++)
	  if ((a[i] != b[i]))
	  {
		return false;
	  }
	return true;

  }



  static void init_matrix(ublas::matrix<T, F>& m, int max_value)
  {
	for (int i = 0; i < m.size1(); i++)
	{
	  for (int j = 0; j<m.size2(); j++)
		m(i, j) = (std::rand() % max_value) + 1;

	}
  }


  static void init_vector(ublas::vector<T>& v, int max_value)
  {
	for (int i = 0; i <v.size(); i++)
	{
	  v[i] = (std::rand() % max_value) + 1;
	}
  }


  virtual void run()
  {
  }

};

#endif
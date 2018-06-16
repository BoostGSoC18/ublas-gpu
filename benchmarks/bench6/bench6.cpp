#include <sys/types.h>
#include <stdio.h>
#define ENABLE_OPENCL
#include <boost/numeric/ublas/matrix.hpp>
#include <time.h>
#include <math.h>
#include <boost/timer.hpp>
#define DATA_TYPE float

namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;

ublas::matrix<DATA_TYPE> a;
ublas::matrix<DATA_TYPE> b;
ublas::matrix<DATA_TYPE> result;



template <class T>
void init_matrix(ublas::matrix<T>& m, int max_value)
{
  for (int i = 0; i < m.size1(); i++)
  {
	for (int j = 0; j < m.size2(); j++)
	  m(i, j) = std::rand() % max_value;

  }
}

void header(const char* str, int dimension)
{
  std::cout << str << dimension << "x" << dimension << std::endl;
}

void init_all(int dimension)
{
  a.resize(dimension, dimension);
  b.resize(dimension, dimension);
  init_matrix(a, 200);
  init_matrix(b, 200);
}


void bench_prod_with_copying(int dimension,  compute::command_queue& queue)
{
  init_all(dimension);

  header("opnecl prod (with copying to host): ", dimension);
  boost::timer t1;
  opencl::prod(a, b, queue);
  std::cout << "time taken = " << t1.elapsed() << " s" << std::endl;

  std::cout << std::endl;

}


void bench_prod_without_copying(int dimension, compute::command_queue& queue)
{
  init_all(dimension);


  ublas::matrix<DATA_TYPE, ublas::basic_row_major<>, opencl::storage> aHolder(a.size1(), a.size2(), queue.get_context());
  aHolder.to_host(a, queue);

  ublas::matrix<DATA_TYPE, ublas::basic_row_major<>, opencl::storage> bHolder(b.size1(), b.size2(), queue.get_context());
  aHolder.to_host(b, queue);

  ublas::matrix<DATA_TYPE, ublas::basic_row_major<>, opencl::storage> resultHolder(a.size1(), b.size2(), queue.get_context());


  header("opnecl prod (without copying to host): ", dimension);
  boost::timer t1;
  opencl::prod(aHolder, bHolder, resultHolder, queue);
  std::cout << "time taken = " << t1.elapsed() << " s" << std::endl;

  std::cout << std::endl;

}




int main()
{
  opencl::library lib;
  std::srand(time(0));

  // get default device and setup context
  compute::device device = compute::system::devices().at(0);
  compute::context context(device);
  compute::command_queue queue(context, device);

  bench_prod_with_copying(100, queue);
  bench_prod_without_copying(100, queue);


  bench_prod_with_copying(200, queue);
  bench_prod_without_copying(200, queue);


  bench_prod_with_copying(500, queue);
  bench_prod_without_copying(500, queue);



  bench_prod_with_copying(1000, queue);
  bench_prod_without_copying(1000, queue);



  bench_prod_with_copying(1500, queue);
  bench_prod_without_copying(1500, queue);


  bench_prod_with_copying(2000, queue);
  bench_prod_without_copying(2000, queue);


  return 0;
}
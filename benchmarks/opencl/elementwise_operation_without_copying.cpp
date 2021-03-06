#define BOOST_UBLAS_ENABLE_OPENCL

#include <boost/numeric/ublas/matrix.hpp>
#include "../benchmark.hpp"

namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;
namespace benchmark = ublas::benchmark;


namespace boost { namespace numeric { namespace ublas { namespace benchmark {

template <typename T, typename L>
class elementwise_operations_opencl_no_copying : public benchmark
{
public:
  elementwise_operations_opencl_no_copying() : benchmark("elementwise operations opencl without copying data") {}
  virtual void setup(long l)
  {
	compute::device device = compute::system::default_device();
	compute::context context(device);
	queue = compute::command_queue(context, device);

	init(a, l, 200);
	init(b, l, 200);
	init(c, l, 200);
  }
  virtual void operation(long l)
  {
	opencl::element_add(a, b, c, queue);
  }
private:


  void init(ublas::matrix<T, L, opencl::storage>&m, long size, int max_value)
  {
	new (&m) ublas::matrix<T, L, opencl::storage>(size, size, max_value, queue);      // Call the constructor 
  }

  ublas::matrix<T, L, opencl::storage> a;
  ublas::matrix<T, L, opencl::storage> b;
  ublas::matrix<T, L, opencl::storage> c;
  compute::command_queue queue;
  opencl::library lib;
};

}}}}



int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 });

  benchmark::elementwise_operations_opencl_no_copying<float, ublas::basic_row_major<>> e1;
  e1.run(times);

}


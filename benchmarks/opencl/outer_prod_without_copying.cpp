#define BOOST_UBLAS_ENABLE_OPENCL

#include <boost/numeric/ublas/matrix.hpp>
#include "../benchmark.hpp"

namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;

namespace boost { namespace numeric { namespace ublas { namespace benchmark {

template <typename T, typename L>
class outer_prod_opencl_no_copying : public benchmark
{
public:
  outer_prod_opencl_no_copying() : benchmark("outer_prod opencl without copying data") {}
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
	opencl::outer_prod(a, b, c, queue);
  }
private:


  void init(ublas::vector<T, opencl::storage>&v, long size, int max_value)
  {
	new (&v) ublas::vector<T, opencl::storage>(size, max_value, queue);      // Call the constructor 
  }

  void init(ublas::matrix<T, L, opencl::storage>&m, long size, int max_value)
  {
	new (&m) ublas::matrix<T, L, opencl::storage>(size, size, queue.get_context());      // Call the constructor 
  }

  ublas::vector<T, opencl::storage> a;
  ublas::vector<T, opencl::storage> b;
  ublas::matrix<T, L, opencl::storage> c;
  compute::command_queue queue;
  opencl::library lib;
};

}}}}


int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 });

  benchmark::outer_prod_opencl_no_copying<float, ublas::basic_row_major<>> o1;
  o1.run(times);
}

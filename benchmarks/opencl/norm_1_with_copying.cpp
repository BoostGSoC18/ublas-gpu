#define BOOST_UBLAS_ENABLE_OPENCL

#include <boost/numeric/ublas/matrix.hpp>
#include "../benchmark.hpp"

namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;
namespace benchmark = ublas::benchmark;


namespace boost { namespace numeric { namespace ublas { namespace benchmark {

template <typename T>
class norm_1_opencl_copying : public benchmark
{
  using vector = ublas::vector<T>;
public:
  norm_1_opencl_copying() : benchmark("norm_1 opencl with copying data") {}
  virtual void setup(long l)
  {
	compute::device device = compute::system::default_device();
	compute::context context(device);
	queue = compute::command_queue(context, device);

	init(a, l, 200);

  }
  virtual void operation(long l)
  {
	opencl::norm_1(a, queue);
  }
private:
  static void init(vector &v, long size, int max_value)
  {
	v = vector(size);
	for (int i = 0; i < v.size(); i++)
	{
	  v[i] = std::rand() % max_value;
	}
  }
  ublas::vector<T> a;
  compute::command_queue queue;
  opencl::library lib;
};

}}}}


int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 });

  benchmark::norm_1_opencl_copying<float> n1;
  n1.run(times);

}

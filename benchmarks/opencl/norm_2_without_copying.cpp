#define BOOST_UBLAS_ENABLE_OPENCL

#include <boost/numeric/ublas/matrix.hpp>
#include "../benchmark.hpp"

namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;

template <typename T>
class norm_2_opencl_no_copying : public benchmark
{
public:
  norm_2_opencl_no_copying() : benchmark("norm_2 opencl without copying data") {}
  virtual void setup(long l)
  {
	compute::device device = compute::system::default_device();
	compute::context context(device);
	queue = compute::command_queue(context, device);

	init(a, l, 200);
  }
  virtual void operation(long l)
  {
	opencl::norm_2(a, queue);
  }
private:


  void init(ublas::vector<T, opencl::storage>&v, long size, int max_value)
  {
	new (&v) ublas::vector<T, opencl::storage>(size, max_value, queue);      // Call the constructor 
  }


  ublas::vector<T, opencl::storage> a;
  compute::command_queue queue;
  opencl::library lib;
};



int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 });

  norm_2_opencl_no_copying<float> n1;
  n1.run(times);

}

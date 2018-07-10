#define ENABLE_OPENCL

#include <boost/numeric/ublas/matrix.hpp>
#include "../benchmark.hpp"

namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;


template <typename T>
class inner_prod_opencl_copying : public benchmark
{
  using vector = ublas::vector<T>;
public:
  inner_prod_opencl_copying() : benchmark("inner_prod opencl with copying data") {}
  virtual void setup(long l)
  {
	compute::device device = compute::system::default_device();
	compute::context context(device);
	queue = compute::command_queue(context, device);

	init(a, l, 200);
	init(b, l, 200);

  }
  virtual void operation(long l)
  {
	opencl::inner_prod(a, b, T(0), queue);
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
  ublas::vector<T> b;
  compute::command_queue queue;
  opencl::library lib;
};


int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131702, 262144 });

  inner_prod_opencl_copying<float> i1;
  i1.run(times);

}

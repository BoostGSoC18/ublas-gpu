#define BOOST_UBLAS_ENABLE_OPENCL

#include <boost/numeric/ublas/matrix.hpp>
#include "../benchmark.hpp"

namespace ublas = boost::numeric::ublas;
namespace opencl = boost::numeric::ublas::opencl;
namespace compute = boost::compute;

template <typename T, typename L>
class elementwise_operations_opencl_copying : public benchmark
{
public:
  elementwise_operations_opencl_copying() : benchmark("elementwise operations opencl with copying data") {}
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
	opencl::element_prod(a, b, queue);
  }
private:
  static void init(ublas::matrix<T, L> &m, long size, int max_value)
  {
	m = ublas::matrix<T, L>(size, size);
	for (int i = 0; i < m.size1(); i++)
	{
	  for (int j = 0; j<m.size2(); j++)
		m(i, j) = std::rand() % max_value;
	}
  }
  ublas::matrix<T, L> a;
  ublas::matrix<T, L> b;
  compute::command_queue queue;
  opencl::library lib;
};



int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 });

  elementwise_operations_opencl_copying<float, ublas::basic_row_major<>> e1;
  e1.run(times);

}

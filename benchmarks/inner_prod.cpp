#include <boost/numeric/ublas/matrix.hpp>
#include "benchmark.hpp"

namespace ublas = boost::numeric::ublas;

template <typename T>
class inner_prod_ublas : public benchmark
{
  using vector = ublas::vector<T>;
public:
  inner_prod_ublas() : benchmark("inner_prod ublas") {}
  virtual void setup(long l)
  {
	init(a, l, 200);
	init(b, l, 200);
  }
  virtual void operation(long l)
  {
	T result = ublas::inner_prod(a, b);
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
};


int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131702, 262144 });

  inner_prod_ublas<float> i1;
  i1.run(times);

}

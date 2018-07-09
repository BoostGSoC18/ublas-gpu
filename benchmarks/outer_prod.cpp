#include <boost/numeric/ublas/matrix.hpp>
#include "benchmark.hpp"

namespace ublas = boost::numeric::ublas;

template <typename T, typename L>
class outer_prod_ublas : public benchmark
{
  using vector = ublas::vector<T>;
public:
  outer_prod_ublas() : benchmark("outer_prod ublas") {}
  virtual void setup(long l)
  {
	init(a, l, 200);
	init(b, l, 200);
  }
  virtual void operation(long l)
  {
	c = ublas::outer_prod(a, b);
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
  ublas::matrix<T, L> c;
};



int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 });

  outer_prod_ublas<float, ublas::basic_row_major<>> o1;
  o1.run(times);

}

#include <boost/numeric/ublas/matrix.hpp>
#include "benchmark.hpp"

namespace ublas = boost::numeric::ublas;

template <typename T>
class norm_2_ublas : public benchmark
{
  using vector = ublas::vector<T>;
public:
  norm_2_ublas() : benchmark("norm_2 ublas") {}
  virtual void setup(long l)
  {
	init(a, l, 200);
  }
  virtual void operation(long l)
  {
	T result = ublas::norm_2(a);
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
};



int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 });

  norm_2_ublas<float> n1;
  n1.run(times);

}

#include <boost/numeric/ublas/matrix.hpp>
#include "benchmark.hpp"

namespace ublas = boost::numeric::ublas;

template <typename T, typename L>
class transposition_ublas : public benchmark
{
  using matrix = ublas::matrix<T, L>;
public:
  transposition_ublas() : benchmark("trans ublas") {}
  virtual void setup(long l)
  {
	init(a, l, 200);
  }
  virtual void operation(long l)
  {
	c = ublas::trans(a);
  }
private:
  static void init(matrix &m, long size, int max_value)
  {
	m = matrix(size, size);
	for (int i = 0; i < m.size1(); i++)
	{
	  for (int j = 0; j<m.size2(); j++)
		m(i, j) = std::rand() % max_value;
	}
  }
  ublas::matrix<T, L> a;
  ublas::matrix<T, L> c;
};


int main(int, char **)
{
  std::vector<long> times({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 });


  transposition_ublas<float, ublas::basic_row_major<>> t1;
  t1.run(times);

}

#include "test_opencl_prod.hpp"
#include <boost/numeric/ublas/matrix.hpp>

int main()
{
	bench_prod<float, ublas::basic_row_major<>, 100, 100> b1;
	bench_prod<double, ublas::basic_row_major<>, 100, 100> b2;

	bench_prod<float, ublas::basic_column_major<>, 100, 100> b3;
	bench_prod<double, ublas::basic_column_major<>, 100, 100> b4;

	std::cout << "Row major:" << std::endl;
	b1.run();
	b2.run();

	std::cout << "Column major:" << std::endl;
	b3.run();
	b4.run();

	return 0;
}
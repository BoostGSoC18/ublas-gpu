#include "prod_test.hpp"
#include <boost/numeric/ublas/matrix.hpp>

int main()
{

	///testing row major flaot prod
	bench_prod<float, ublas::basic_row_major<>, 10, 10> b1;

	///testing row major complex float prod
	bench_prod<std::complex<float>, ublas::basic_row_major<>, 10, 10> b2;

	///testing column major flaot prod
	bench_prod<float, ublas::basic_column_major<>, 10, 10> b3;

	///testing column major complex float prod
	bench_prod<std::complex<float>, ublas::basic_column_major<>, 10, 10> b4;

	std::cout << "Row major:" << std::endl;
	b1.run();
	b2.run();

	std::cout << "Column major:" << std::endl;
	b3.run();
	b4.run();

		return 0;

}
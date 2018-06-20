#include "elementwise_operations_test.hpp"

int main()
{

  ///testing row major flaot elementwise operations
  bench_elementwise <float, ublas::basic_row_major<>, 10, 10> b1;

  ///testing row major complex float elementwise operations
  bench_elementwise <std::complex<float>, ublas::basic_row_major<>, 10, 10> b2;

  ///testing column major flaot elementwise operations
  bench_elementwise <float, ublas::basic_column_major<>, 10, 10> b3;
 
  ///testing column major complex float elementwise operations
  bench_elementwise <std::complex<float>, ublas::basic_column_major<>, 10, 10> b4;

  std::cout << "row major:" << std::endl;
  b1.run();
  b2.run();
 
  std::cout << "column major:" << std::endl;
  b3.run();
  b4.run();
 
}

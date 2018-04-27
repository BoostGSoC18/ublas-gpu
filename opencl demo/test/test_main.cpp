#include "test_opencl_prod.hpp"

int main()
{
	bench_prod<float, 100, 100> b1;
	bench_prod<double, 100, 100> b2;

	b1.run();
	b2.run();

	return 0;
}
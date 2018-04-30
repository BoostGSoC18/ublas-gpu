#include <sys/types.h>
#include <stdio.h>
#define ENABLE_OPENCL
#include <boost/numeric/ublas/matrix.hpp>
#include <time.h>
#include <math.h>
#include <boost/timer.hpp>
#define DATA_TYPE float

namespace ublas = boost::numeric::ublas;

ublas::matrix<DATA_TYPE> a;
ublas::matrix<DATA_TYPE> b;
ublas::matrix<DATA_TYPE> result;


void run_prod_opencl(int number_of_times, opencl::opencl_device& device)
{
	for (int i = 0; i < number_of_times; i++)
	{
		result = opencl::prod(a, b, device);
	}
}

void run_prod_ublas(int number_of_times)
{
	for (int i = 0; i < number_of_times; i++)
	{
		result = prod(a, b);
	}
}

template <class T>
void init_matrix(ublas::matrix<T>& m, int max_value)
{
	for (int i = 0; i < m.size1(); i++)
	{
		for (int j = 0; j<m.size2(); j++)
			m(i, j) = std::rand() % max_value;

	}
}

void header(char* str, int dimension)
{
	std::cout << str << dimension <<"x"<< dimension<< std::endl;
}

void init_all(int dimension)
{
	a.resize(dimension, dimension);
	b.resize(dimension, dimension);
	init_matrix(a, 200);
	init_matrix(b, 200);
}


void bench_difference(int dimension,int times, opencl::opencl_device& device)
{
	init_all(dimension);

	header("opnecl prod : ", dimension);
	boost::timer t1;
	run_prod_opencl(times, device);
	std::cout << "time taken = " << t1.elapsed() << " s"<<std::endl;


	// header("ublas prod : ", dimension);
	// boost::timer t0;
	// run_prod_ublas(times);
	// std::cout << "time taken = " << t0.elapsed() << " s" << std::endl;


	std::cout << std::endl;

}

int main()
{

	std::srand(time(0));

	opencl::startOpencl();
	opencl::opencl_device device(2);
	
	bench_difference(100, 1, device);
	bench_difference(200, 1, device);
	bench_difference(500, 1, device);
	bench_difference(1000, 1, device);
	bench_difference(1500, 1, device);



	opencl::endOpencl();
	return 0;
}
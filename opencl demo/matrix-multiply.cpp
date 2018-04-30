#include <sys/types.h>
#include <stdio.h>
#define ENABLE_OPENCL
#include <boost/numeric/ublas/matrix.hpp>
#include <time.h>

/* This example uses predefined matrices and their characteristics for
* simplicity purpose.
*/

namespace compute = boost::compute;
namespace ublas = boost::numeric::ublas;

int main(void)
{

	opencl::opencl_device device(2);

	int size = 500;
	ublas::matrix<float> a(size, size, 63);

	ublas::matrix<float> b(size, size, 12);

	ublas::matrix<float> resU(size, size, 3);
	clock_t start, end;



	opencl::startOpencl(); //2 is a device number of the user's choice

	start = clock();
	resU = opencl::prod(a, b, device);
	end = clock();

	opencl::endOpencl();



	/*for (int i = 0; i < resU.size1(); i++)
	{
	for (int j = 0; j < resU.size2(); j++)
	std::cout << resU(i,j) << "    ";

	std::cout << std::endl;
	}
	*/




	std::cout << "time on gpu = " << end - start << std::endl;

	start = clock();
	resU = prod(a, b);
	end = clock();

	std::cout << "time on cpu = " << end - start << std::endl;

	return 0;

}

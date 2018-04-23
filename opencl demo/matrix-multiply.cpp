#include <sys/types.h>
#include <stdio.h>
#include <boost\compute\algorithm.hpp>
#include <boost\numeric\ublas\matrix.hpp>
#include <time.h>
/* Include the clBLAS header. It includes the appropriate OpenCL headers */
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
* simplicity purpose.
*/

namespace compute = boost::compute;
namespace ublas = boost::numeric::ublas;

int main(void)
{

	// get default device and setup context
	compute::device device = compute::system::devices().at(2);
	compute::context context(device);
	compute::command_queue queue(context, device);

	cl_int err;
	int size = 1000;
	ublas::matrix<float> a(size, size, 63);
	compute::vector<float> aC(size*size,context);

	ublas::matrix<float> b(size, size, 12);
	compute::vector<float> bC(size*size,context);

	ublas::matrix<float> resU(size, size,3);
	compute::vector<float> res(size*size, context);
	clock_t start, end;

/* Setup clBLAS */
	err = clblasSetup();
	cl_event event = NULL;


	start = clock();

	compute::copy(
		a.data().begin(),
		a.data().end(),
		aC.begin(),
		queue
	);

	compute::copy(
		b.data().begin(),
		b.data().end(),
		bC.begin(),
		queue
	);

	

	/* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
	err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
		size, size, size,
		1, (cl_mem)aC.begin().get_buffer().get(), 0, size,
		(cl_mem)bC.begin().get_buffer().get(), 0, size,1,
		(cl_mem)res.begin().get_buffer().get(), 0, size,
		1,&(queue.get()), 0, NULL, &event);





	/* Wait for calculations to be finished. */
	err = clWaitForEvents(1, &event);

	queue.finish();


	compute::copy(
		res.begin(),
		res.end(),
		resU.data().begin(),
		queue
	);

	// for (int i = 0; i < resU.size1(); i++)
	// {
	// 	for (int j = 0; j < resU.size2(); j++)
	// 		std::cout << resU(i,j) << "    ";

	// 	std::cout << std::endl;
	// }


	
	end = clock();

	/* Finalize work with clBLAS */
	clblasTeardown();

	std::cout << "time on gpu = " << end - start<<std::endl;

	start = clock();
	resU = prod(a,b);
	end = clock();

	std::cout << "time on cpu = " << end - start << std::endl;

	return 0;

}

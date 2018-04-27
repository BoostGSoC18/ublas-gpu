#include <sys/types.h>
#include <stdio.h>
#define ENABLE_OPENCL
#include <boost/numeric/ublas/matrix.hpp>
#include <time.h>
#include <math.h>

namespace ublas = boost::numeric::ublas;

#define NUMBER_OF_TESTS 100
#define MAX_DIMENSION 100
#define DATA_TYPE float


template <class T>
bool compare(ublas::matrix<T>& a , ublas::matrix<T>& b)
{
	if( (a.size1() != b.size1()) || (a.size2() != b.size2()) )
		return false;

	for(int i=0; i<a.size1(); i++)
		for(int j=0; j<a.size2(); j++)
			if(a(i,j) != b(i,j) )
					return false;

	return true;

}

template <class T>
void init_matrix(ublas::matrix<T>& m, int max_value)
{
	for (int i = 0; i < m.size1(); i++)
	{
		for(int j=0; j<m.size2(); j++)
			m(i,j) = std::rand()%max_value;
		
	}
}

int main()
{

	int passedOperations =0;
	opencl::opencl_device device(2);
    std::srand(time(0));

	opencl::startOpencl();

	ublas::matrix<DATA_TYPE> a;
	ublas::matrix<DATA_TYPE> b;
	ublas::matrix<DATA_TYPE> resultUBLAS;
	ublas::matrix<DATA_TYPE> resultOPENCL;


	for(int i=0; i<NUMBER_OF_TESTS; i++)
	{
		int rowsA = std::rand()%MAX_DIMENSION+1;
		int colsA = std::rand()%MAX_DIMENSION +1;
		int colsB = std::rand()%MAX_DIMENSION +1;
		
		 a.resize(rowsA,colsA);
		 b.resize(colsA, colsB);

		
		init_matrix(a,200);
		init_matrix(b,200);


		 resultUBLAS = prod(a,b);
		 resultOPENCL = opencl::prod(a,b, device);


		if(!compare(resultUBLAS,resultOPENCL))
		{
			std::cout<< "Error in calculations"<<std::endl;

			std::cout<<"passed: "<< passedOperations <<std::endl;
			return 0;
		}

		 passedOperations++;

	}
		std::cout<<"All is well"<<std::endl;
		opencl::endOpencl();



	return 0;
}
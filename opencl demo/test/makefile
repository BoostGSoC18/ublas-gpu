CC=g++
Deps= test_opencl_prod.cpp
OutputName=test
boost_include_dir="C:/Program Files/boost/boost_1_66_0"
opencl_include_dir="C:/Program Files (x86)/AMD APP SDK/3.0/include"
clblas_include_dir="C:/Program Files/clBLAS/src"


all: $(Deps) #first make command to build 
	$(CC) $(Deps) -I$(boost_include_dir) -I$(opencl_include_dir) -I$(clblas_include_dir) -lclBLAS -lOpenCL -o $(OutputName)


run: $(OutputName)
	./$(OutputName)

clean: $(OutputName) 
	rm $(OutputName)
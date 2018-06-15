#ifndef OPENCL_MATRIX
#define OPENCL_MATRIX

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/compute/buffer.hpp>

/// Include the clBLAS header. It includes the appropriate OpenCL headers
#include <clBLAS.h>

namespace boost {
	namespace numeric {
		namespace ublas {


			namespace opencl
			{
				namespace compute = boost::compute;
				namespace ublas = boost::numeric::ublas;


				class storage;

				class opencl_device
				{
				private:
					compute::device device;
					compute::context context;
					compute::command_queue queue;

				public:
					opencl_device() {}

					opencl_device(unsigned int device_number)
					{
						this->device = compute::system::devices().at(device_number);
						this->context = compute::context(device);
						this->queue = compute::command_queue(context, device);
					}


					void initialize(unsigned int device_number)
					{
						this->device = compute::system::devices().at(device_number);
						this->context = compute::context(device);
						this->queue = compute::command_queue(context, device);
					}

					compute::device& getDevice()
					{
						return device;
					}


					const compute::device& getDevice() const
					{
						return device;
					}

					compute::context& getContext()
					{
						return context;
					}

					const compute::context& getContext() const
					{
						return context;
					}

					const compute::command_queue& getQueue() const
					{
						return queue;
					}

					compute::command_queue& getQueue()
					{
						return queue;
					}



				};

				inline void startOpencl()
				{
					///start clBlas
					clblasSetup();

				}

				inline void endOpencl()
				{
					/// Finalize clBLAS

					clblasTeardown();

				}

				///copy data from ublas matrix to gpu vector
				template <class T, class F, class A>
				void copy_to_device(ublas::matrix<T, F, A>& m, ublas::matrix<T, F, opencl::storage>& mGPU, opencl::opencl_device device)
				{
					compute::copy(
						m.data().begin(),
						m.data().end(),
						mGPU.data().begin(),
						device.getQueue()
					);

				}

				///copy data from gpu vector to ublas matrix
				template <class T, class F, class A>
				void copy_from_device(ublas::matrix<T, F, A>& m, ublas::matrix<T, F, opencl::storage>& mGPU)
				{
					compute::copy(
						mGPU.data().begin(),
						mGPU.data().end(),
						m.data().begin(),
						mGPU.device().getQueue()
					);

				}



			}//opencl

			 /**
			 matrix class to keep data on gpu in case we need to do further computations on them
			 */
			template<class T, class L>
			class matrix<T, L, opencl::storage> : public matrix_container<matrix<T, L, opencl::storage> >

			{
				typedef typename boost::compute::buffer_allocator<T>::size_type size_type;
				typedef L layout_type;
				typedef matrix<T, L, opencl::storage> self_type;

			public:
				matrix(size_type size1, size_type size2, opencl::opencl_device const &d)
					: matrix_container<self_type>(),
					size1_(size1), size2_(size2), data_(compute::vector<T>(layout_type::storage_size(size1, size2), d.getContext())), device_(d)
				{}

				matrix(size_type size1, size_type size2, opencl::opencl_device &d, const T& value)
					: matrix_container<self_type>(),
					size1_(size1), size2_(size2), data_(compute::vector<T>(layout_type::storage_size(size1, size2), d.getContext())), device_(d)
				{
					compute::fill(data_.begin(), data_.end(), value, d.getQueue());
				}


				matrix(opencl::opencl_device& d)
					: matrix_container<self_type>(),
					size1_(0), size2_(0), data_(compute::vector<T>(d.getContext())), device_(d)
				{}

				// Accessors
				/** Return the number of rows of the matrix
				*/
				BOOST_UBLAS_INLINE
					size_type size1() const {
					return size1_;
				}

				/** Return the number of colums of the matrix
				*/
				BOOST_UBLAS_INLINE
					size_type size2() const {
					return size2_;
				}

				/** Return the compute::vector that has the data on gpu
				*/
				const compute::vector<T> &data() const { return data_; }
				compute::vector<T> &data() { return data_; }

				/** Return the device that the matrix resides on
				*/
				const opencl::opencl_device &device() const { return device_; }
				opencl::opencl_device &device() { return device_; }


				/** Resize the matrix
				*/
				void resize(size_type size1, size_type size2)
				{
					size1_ = size1;
					size2_ = size2;
					data_.resize(layout_type::storage_size(size1, size2));
				}

				/**
				Fill all elements of the matrix with the value
				*/
				void fill(T value)
				{
					compute::fill(data_.begin(), data_.end(), value, device_.getQueue());
				}


			private:
				size_type size1_;
				size_type size2_;
				compute::vector<T> data_;
				opencl::opencl_device device_;
			};
		}//ublas
	}//numeric
}//boost


#endif 
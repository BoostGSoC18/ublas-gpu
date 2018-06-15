#ifndef OPENCL_MATRIX
#ifndef OPENCL_MATRIX
#define OPENCL_MATRIX

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/compute/buffer.hpp>
#include <clBLAS.h>

namespace boost {
  namespace numeric {
	namespace ublas {


	  namespace opencl
	  {
		namespace compute = boost::compute;
		namespace ublas = boost::numeric::ublas;

		/// a class that is used as a tag to tell that the matrix resides on a device
		class storage;


		/** \breif this class is used to store information about a specific device
		*
		* the opencl library takes some overhead to get information about a devic and get it's context
		* and command queue so we use this class to store these information for later accesing without consuming time.
		*
		*/
		class opencl_device
		{

		public:
		  //Construction

		  /** initialize empty opencl_device
		  */
		  BOOST_UBLAS_INLINE
			opencl_device() {}


		  /** constructs opencl_device with information about device number device_number on the system
		  * \param device_number the number of the device on the system
		  */
		  BOOST_UBLAS_INLINE
			opencl_device(unsigned int device_number)
		  {
			this->device = compute::system::devices().at(device_number);
			this->context = compute::context(device);
			this->queue = compute::command_queue(context, device);
		  }

		  /** initialize opencl_device to have information about device number device_number on the system
		  * \param device_number the number of the device on the system
		  */
		  BOOST_UBLAS_INLINE
			void initialize(unsigned int device_number)
		  {
			this->device = compute::system::devices().at(device_number);
			this->context = compute::context(device);
			this->queue = compute::command_queue(context, device);
		  }

		  ///returns boost::compute::device that has information about the device
		  BOOST_UBLAS_INLINE
			compute::device& getDevice()
		  {
			return device;
		  }

		  ///returns boost::compute::device that has information about the device
		  BOOST_UBLAS_INLINE
			const compute::device& getDevice() const
		  {
			return device;
		  }

		  ///returns boost::compute::context that has information about the context of the device
		  BOOST_UBLAS_INLINE
			compute::context& getContext()
		  {
			return context;
		  }

		  ///returns boost::compute::context that has information about the context of the device
		  BOOST_UBLAS_INLINE
			const compute::context& getContext() const
		  {
			return context;
		  }

		  ///returns boost::compute::command_queue that has information about the command queue of the device
		  BOOST_UBLAS_INLINE
			const compute::command_queue& getQueue() const
		  {
			return queue;
		  }

		  ///returns boost::compute::command_queue that has information about the command queue of the device
		  BOOST_UBLAS_INLINE
			compute::command_queue& getQueue()
		  {
			return queue;
		  }

		private:
		  compute::device device;
		  compute::context context;
		  compute::command_queue queue;

		};


		///start clBlas (used for the matrix product kernel
		BOOST_UBLAS_INLINE
		  inline void startOpencl()
		{
		  clblasSetup();
		}

		/// Finalize clBLAS (used for the matrix product kernel)
		BOOST_UBLAS_INLINE
		  inline void endOpencl()
		{
		  clblasTeardown();
		}






	  }//opencl

	   /** \brief A dense matrix of values of type \c T that resides on a specific device using opencl.
	   *
	   * this is a special case of class boost::numeric::ublas::matrix<T, L, A> that has A as opencl::storage
	   * so it is stored on a specific device and stores some information about this device.
	   *
	   *
	   * For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
	   * the \f$(i.n + j)\f$-th element of the container for row major orientation or the \f$ (i + j.m) \f$-th element of
	   * the container for column major orientation.
	   *
	   *
	   * Orientation and storage can also be specified, otherwise a \c row_major and \c unbounded_array are used. It is \b not
	   * required by the storage to initialize elements of the matrix.
	   *
	   * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
	   * \tparam L the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
	   */
	  template<class T, class L>
	  class matrix<T, L, opencl::storage> : public matrix_container<matrix<T, L, opencl::storage> >

	  {
		typedef typename boost::compute::buffer_allocator<T>::size_type size_type;
		typedef L layout_type;
		typedef matrix<T, L, opencl::storage> self_type;

	  public:

		// Construction


		/** Dense matrix constructor with size (0,0)
		* \param d is the boost::nemeric::ublas::opencl::opencl_device that contains the matrix on its memory
		*/
		BOOST_UBLAS_INLINE
		  matrix(opencl::opencl_device& d)
		  : matrix_container<self_type>(),
		  size1_(0), size2_(0), data_(compute::vector<T>(d.getContext())), device_(d)
		{}

		/** Dense matrix constructor with size (size1,size2) and resides on memory of device d
		* \param d is the boost::nemeric::ublas::opencl::opencl_device that contains the matrix on its memory
		* \param size1 number of rows
		* \param size2 number of columns
		*/
		BOOST_UBLAS_INLINE
		  matrix(size_type size1, size_type size2, opencl::opencl_device &d)
		  : matrix_container<self_type>(),
		  size1_(size1), size2_(size2), data_(compute::vector<T>(layout_type::storage_size(size1, size2), d.getContext())), device_(d)
		{}


		/** Dense matrix constructor with size (size1,size2) and resides on memory of device d and initialize all elements to value
		* \param d is the boost::nemeric::ublas::opencl::opencl_device that contains the matrix on its memory
		* \param size1 number of rows
		* \param size2 number of columns
		* \param value is the value that all elements of the matrix are set to
		*/
		BOOST_UBLAS_INLINE
		  matrix(size_type size1, size_type size2, opencl::opencl_device &d, const T& value)
		  : matrix_container<self_type>(),
		  size1_(size1), size2_(size2), data_(compute::vector<T>(layout_type::storage_size(size1, size2), d.getContext())), device_(d)
		{
		  compute::fill(data_.begin(), data_.end(), value, d.getQueue());
		}


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

		/** Return the compute::vector that has the data on the device
		*/
		BOOST_UBLAS_INLINE
		  const compute::vector<T> &data() const { return data_; }

		BOOST_UBLAS_INLINE
		  compute::vector<T> &data() { return data_; }

		/** Return boost::numeri::ublas::opencl::opencl_device that has informaton
		* about the device that the matrix resides on.
		*/
		BOOST_UBLAS_INLINE
		  const opencl::opencl_device &device() const { return device_; }

		BOOST_UBLAS_INLINE
		  opencl::opencl_device &device() { return device_; }


		/** Resize the matrix to specific size
		* \param size1 new number of rows
		* \param size2 new number of columns
		*/
		BOOST_UBLAS_INLINE
		  void resize(size_type size1, size_type size2)
		{
		  size1_ = size1;
		  size2_ = size2;
		  data_.resize(layout_type::storage_size(size1, size2));
		}

		/**Fill all elements of the matrix with the value
		* \param value value to set all elements of the matrix to
		*/
		BOOST_UBLAS_INLINE
		  void fill(T value)
		{
		  compute::fill(data_.begin(), data_.end(), value, device_.getQueue());
		}

		/** Copies a matrix to a device
		* \param m is a matrix that is not on the device _device and it is copied to it
		*/
		BOOST_UBLAS_INLINE
		  template<class A>
		void to_host(ublas::matrix<T, L, A>& m)
		{
		  this->resize(m.size1(), m.size2());
		  compute::copy(
			m.data().begin(),
			m.data().end(),
			this->data().begin(),
			device_.getQueue()
		  );
		}


		/** Copies a matrix from a device
		* \param m is a matrix that will be reized to (size1_,size2) and the values of (*this) will be copied in it
		*/
		BOOST_UBLAS_INLINE
		  template<class A>
		void from_host(ublas::matrix<T, L, A>& m)
		{
		  m.resize(size1_, size2_);
		  compute::copy(
			this->data().begin(),
			this->data().end(),
			m.data().begin(),
			device_.getQueue()
		  );
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
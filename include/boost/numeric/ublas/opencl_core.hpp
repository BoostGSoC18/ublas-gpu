#ifndef OPENCL_MATRIX
#define OPENCL_MATRIX

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/compute/buffer.hpp>
#include <boost/compute/container/vector.hpp>
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


  ///a class to initialize and finalize library
  class library
  {

  public:

	///start clBlas (used for the matrix product kernel
	library()
	{
	  clblasSetup();
	}

	/// Finalize clBLAS (used for the matrix product kernel)
	~library()
	{
	  clblasTeardown();
	}

  };




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

  /** Dense matrix constructor with size (size1,size2) and resides on memory of device of queue q
  * \param q is the boost::compute::command_queue that contains the matrix on its device memory
  * \param size1 number of rows
  * \param size2 number of columns
  */

  matrix(size_type size1, size_type size2, compute::command_queue &q)
	: matrix_container<self_type>(),
	size1_(size1), size2_(size2), queue_(q)
  {
	compute::buffer_allocator<T> allocator(q.get_context());
	data_ = allocator.allocate(layout_type::storage_size(size1, size2)).get_buffer();
  }


  /** Dense matrix constructor with size (size1,size2) and resides on memory of device of queue q and initialize all elements to value
  * \param 2 is the boost::compute::command_queue that contains the matrix on its device memory
  * \param size1 number of rows
  * \param size2 number of columns
  * \param value is the value that all elements of the matrix are set to
  */

  matrix(size_type size1, size_type size2, compute::command_queue &q, const T& value)
	: matrix_container<self_type>(),
	size1_(size1), size2_(size2), queue_(q)
  {
	compute::buffer_allocator<T> allocator(q.get_context());
	data_ = allocator.allocate(layout_type::storage_size(size1, size2)).get_buffer();

	compute::fill(this->begin(), this->end(), value, q);
  }


  // Accessors
  /** Return the number of rows of the matrix
  */

  size_type size1() const {
	return size1_;
  }

  /** Return the number of colums of the matrix
  */

  size_type size2() const {
	return size2_;
  }

  /** Return the compute::buffer that has the begin data on the device
  */

  const compute::buffer_iterator<T> begin() const { compute::make_buffer_iterator<T>(data_); }
  compute::buffer_iterator<T> begin() { return compute::make_buffer_iterator<T>(data_); }


  /** Return the compute::buffer that has the end of the data on the device
  */
  compute::buffer_iterator<T> end() { return compute::make_buffer_iterator<T>(data_, layout_type::storage_size(size1_, size2_)); }
  const compute::buffer_iterator<T> end() const { return compute::make_buffer_iterator<T>(data_, layout_type::storage_size(size1_, size2_)); }

  /** Return boost::numeri::ublas::opencl::queue that has informaton
  * about the device that the matrix resides on.
  */

  const compute::command_queue &queue() const { return queue_; }


  compute::command_queue &queue() { return queue_; }


  /**Fill all elements of the matrix with the value
  * \param value value to set all elements of the matrix to
  */

  void fill(T value)
  {
	compute::fill(this->begin(), this->end(), value, queue_);
  }

  /** Copies a matrix to a device
  * \param m is a matrix that is not on the device _device and it is copied to it
  */

  template<class A>
  void to_host(ublas::matrix<T, L, A>& m)
  {
	compute::copy(
	  m.data().begin(),
	  m.data().end(),
	  this->begin(),
	  queue_
	);
  }


  /** Copies a matrix from a device
  * \param m is a matrix that will be reized to (size1_,size2) and the values of (*this) will be copied in it
  */

  template<class A>
  void from_host(ublas::matrix<T, L, A>& m)
  {
	compute::copy(
	  this->begin(),
	  this->end(),
	  m.data().begin(),
	  queue_
	);
  }


private:
  size_type size1_;
  size_type size2_;
  compute::buffer data_;
  compute::command_queue queue_;
};
}//ublas
}//numeric
}//boost


#endif 
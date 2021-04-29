#ifndef __TYPE_COMPATIBLE_H
#define __TYPE_COMPATIBLE_H

namespace istc
{
/*namespace starts*/

template<typename T> struct SVDReal{};

template <> struct SVDReal<float> { typedef float type; };
  
template <> struct SVDReal<double> { typedef double type; };
  
template <typename T> struct SVDReal<std::complex<T>> 
{ 
    typedef typename SVDReal<T>::type type; 
};


template <typename T> struct isComplex{enum{value = false};};

template <> struct isComplex<std::complex<float>> {enum{value = true};};
template <> struct isComplex<const std::complex<float>> {enum{value = true};};

template <> struct isComplex<std::complex<double>> {enum{value = true};};
template <> struct isComplex<const std::complex<double>> {enum{value = true};};

/*namespace ends*/
}
#endif

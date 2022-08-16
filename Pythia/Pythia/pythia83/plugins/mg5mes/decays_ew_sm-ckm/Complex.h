#ifndef COMPLEX_H_INCLUDED
#define COMPLEX_H_INCLUDED

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

template <class T>
class Complex {
public:
	enum { D = 2 };

	//Complex(const Complex<T> &S) : uni.vals._a(S.uni.vals._a), uni.vals._b(S.uni.vals._b) { }
	Complex(const Complex<T> &S)
          { uni.vals._a = S.uni.vals._a; uni.vals._b = S.uni.vals._b; }
	//Complex(const T U[D]) : a(U[0]), b(U[1]) { }
	Complex(const T U[D])
          { uni.vals._a = U[0]; uni.vals._b =U[1]; }
	//Complex(const T &sa = (T)0, const T &sb = (T)0) : uni.vals._a(sa), uni.vals._b(sb) { }
	Complex(const T &sa = (T)0, const T &sb = (T)0)
          { uni.vals._a = sa; uni.vals._b = sb; }

	const Complex<T> & set(const T &, const T &);

	const Complex<T> operator + (const Complex<T> &) const;
	const Complex<T> operator - (const Complex<T> &) const;
	const Complex<T> operator * (const Complex<T> &) const;
	const Complex<T> operator / (const Complex<T> &) const;
	const Complex<T> operator + (const T &) const;
	const Complex<T> operator - (const T &) const;
	const Complex<T> operator * (const T &) const;
	const Complex<T> operator / (const T &) const;

	const Complex<T> & operator = (const Complex<T> &);
	const Complex<T> & operator += (const Complex<T> &);
	const Complex<T> & operator -= (const Complex<T> &);
	const Complex<T> & operator *= (const Complex<T> &);
	const Complex<T> & operator /= (const Complex<T> &);
	const Complex<T> & operator = (const T &);
	const Complex<T> & operator += (const T &);
	const Complex<T> & operator -= (const T &);
	const Complex<T> & operator *= (const T &);
	const Complex<T> & operator /= (const T &);

	const Complex<T> operator - () const;
	const Complex<T> operator + () const;
	const T & operator [] (const int) const;
	T & operator [] (const int);

        T re() { return uni.vals._a;}
        T im() { return uni.vals._b;}
        T real() { return uni.vals._a;}
        T imag() { return uni.vals._b;}

	std::ofstream & saveBin(std::ofstream &) const;
	std::ifstream & loadBin(std::ifstream &);

	T normalize(void);

	~Complex(void) { }

	union {
          T V[D];
          struct {
            T _a, _b;
          } vals;
        } uni;
};

template <class T>
inline const T abs(const Complex<T> &A)
{
    return sqrt(A.uni.vals._a*A.uni.vals._a + A.uni.vals._b*A.uni.vals._b);
}

template <class T>
inline const T absSq(const Complex<T> &A)
{
    return A.uni.vals._a*A.uni.vals._a + A.uni.vals._b*A.uni.vals._b;
}

template <class T>
inline const T arg(const Complex<T> &A)
{
    return atan2(A.uni.vals._b, A.uni.vals._a);
}

template <class T>
const Complex<T> conj(const Complex<T> &A)
{
    return Complex<T>(A.uni.vals._a, -A.uni.vals._b);
}

template <class T>
inline const T re(const Complex<T> &A)
{
    return A.uni.vals._a;
}

template <class T>
inline const T real(const Complex<T> &A)
{
    return A.uni.vals._a;
}

template <class T>
inline const T im(const Complex<T> &A)
{
    return A.uni.vals._b;
}

template <class T>
inline const T imag(const Complex<T> &A)
{
    return A.uni.vals._b;
}

template <class T>
inline Complex<T> sqrt(const Complex<T> &C)
{
	T l = abs(C);
	T p = 0.5*arg(C);

	return Complex<T>(l*cos(p), l*sin(p));
}

template <class T>
inline Complex<T> exp(const Complex<T> &C)
{
	T k = exp(C.uni.vals._a);
	return Complex<T>(cos(C.uni.vals._b)*k, sin(C.uni.vals._b)*k);
}

template <class T>
inline Complex<T> pow(const Complex<T> &C, const T &m)
{
	T l = pow(abs(C), m);
	T p = m*arg(C);

	return Complex<T>(l*cos(p), l*sin(p));
}

template <class T>
inline Complex<T> pow(const T &a, const Complex<T> &C)
{
	T p = pow(a, C.uni.vals._a);
	T l = log(C.uni.vals._a);

	return Complex<T>(p*cos(C.uni.vals._b*l), p*sin(C.uni.vals._b*l));
}

template <class T>
std::ostream & operator << (std::ostream &vout, const Complex<T> &Q)
{
    return vout << "" << std::setw(14) << Q.uni.vals._a << " + " << std::setw(14) << Q.uni.vals._b << "i";
}

template <class T>
inline const Complex<T> operator + (const T &l, const Complex<T> &R)
{
    return Complex<T>(l + R.uni.vals._a, R.uni.vals._b);
}

template <class T>
inline const Complex<T> operator - (const T &l, const Complex<T> &R)
{
    return Complex<T>(l - R.uni.vals._a, -R.uni.vals._b);
}

template <class T>
inline const Complex<T> operator * (const T &l, const Complex<T> &R)
{
    return Complex<T>(l*R.uni.vals._a, l*R.uni.vals._b);
}

template <class T>
inline const Complex<T> operator / (const T &l, const Complex<T> &R)
{
    T z = absSq(R);
    return Complex<T>(l*R.uni.vals._a/z, -l*R.uni.vals._b/z);
}

template <class T>
inline const Complex<T> & Complex<T>::set(const T &sa, const T &sb)
{
	uni.vals._a = sa; uni.vals._b = sb;
	return *this;
}

template <class T>
inline const Complex<T> Complex<T>::operator + (const Complex<T> &R) const
{
    return Complex<T>(uni.vals._a + R.uni.vals._a, uni.vals._b + R.uni.vals._b);
}

template <class T>
inline const Complex<T> Complex<T>::operator - (const Complex<T> &R) const
{
    return Complex<T>(uni.vals._a - R.uni.vals._a, uni.vals._b - R.uni.vals._b);
}

template <class T>
inline const Complex<T> Complex<T>::operator * (const Complex<T> &R) const
{
    return Complex<T>(uni.vals._a*R.uni.vals._a - uni.vals._b*R.uni.vals._b, uni.vals._a*R.uni.vals._b + uni.vals._b*R.uni.vals._a);
}

template <class T>
inline const Complex<T> Complex<T>::operator / (const Complex<T> &R) const
{
    T z = abs(R);
    return Complex<T>((uni.vals._a*R.uni.vals._a + uni.vals._b*R.uni.vals._b)/z, (uni.vals._b*R.uni.vals._a - uni.vals._a*R.uni.vals._b)/z);
}

template <class T>
inline const Complex<T> Complex<T>::operator + (const T &r) const
{
    return Complex<T>(uni.vals._a + r, uni.vals._b);
}

template <class T>
inline const Complex<T> Complex<T>::operator - (const T &r) const
{
    return Complex<T>(uni.vals._a - r, uni.vals._b);
}

template <class T>
inline const Complex<T> Complex<T>::operator * (const T &r) const
{
    return Complex<T>(uni.vals._a*r, uni.vals._b*r);
}

template <class T>
inline const Complex<T> Complex<T>::operator / (const T &r) const
{
    return Complex<T>(uni.vals._a/r, uni.vals._b/r);
}

template <class T>
inline const Complex<T> & Complex<T>::operator = (const Complex<T> &R)
{
    return set(R.uni.vals._a, R.uni.vals._b);
}

template <class T>
inline const Complex<T> & Complex<T>::operator += (const Complex<T> &R)
{
    return set(uni.vals._a + R.uni.vals._a, uni.vals._b + R.uni.vals._b);
}

template <class T>
inline const Complex<T> & Complex<T>::operator -= (const Complex<T> &R)
{
    return set(uni.vals._a - R.uni.vals._a, uni.vals._b - R.uni.vals._b);
}

template <class T>
inline const Complex<T> & Complex<T>::operator *= (const Complex<T> &R)
{
    return set(uni.vals._a*R.uni.vals._a - uni.vals._a*R.uni.vals._b, uni.vals._a*R.uni.vals._b + uni.vals._b*R.uni.vals._a);
}

template <class T>
inline const Complex<T> & Complex<T>::operator /= (const Complex<T> &R)
{
    T z = abs(R);
    return set((uni.vals._a*R.uni.vals._a + uni.vals._a*R.uni.vals._b)/z, (uni.vals._b*R.uni.vals._a - uni.vals,uni.vals._a*R.uni.vals._b)/z);
}

template <class T>
inline const Complex<T> & Complex<T>::operator = (const T &r)
{
    return set(r, static_cast<T>(0));
}

template <class T>
inline const Complex<T> & Complex<T>::operator += (const T &r)
{
    return set(uni.vals._a + r, uni.vals._b);
}

template <class T>
inline const Complex<T> & Complex<T>::operator -= (const T &r)
{
    return set(uni.vals._a - r, uni.vals._b);
}

template <class T>
inline const Complex<T> & Complex<T>::operator *= (const T &r)
{
    return set(uni.vals._a*r, uni.vals._b*r);
}

template <class T>
inline const Complex<T> & Complex<T>::operator /= (const T &r)
{
    return set(uni.vals._a/r, uni.vals._b/r);
}

template <class T>
inline const Complex <T> Complex<T>::operator - () const
{
	return Complex(-uni.vals._a, -uni.vals._b);
}

template <class T>
inline const Complex <T> Complex<T>::operator + () const
{
	return Complex(uni.vals._a,uni.vals._b);
}

template <class T>
inline const T & Complex<T>::operator [] (const int en) const
{
	return uni.V[en];
}

template <class T>
inline T & Complex<T>::operator [] (const int en)
{
	return uni.V[en];
}

template <class T>
inline std::ofstream & Complex<T>::saveBin(std::ofstream &savf) const
{
	savf.write((char *)uni.V, D*sizeof(T));
	return savf;
}

template <class T>
inline std::ifstream & Complex<T>::loadBin(std::ifstream &loaf)
{
	loaf.read((char *)uni.V, D*sizeof(T));
	return loaf;
}


template <class T>
inline T Complex<T>::normalize(void)
{
	T l = sqrt(uni.vals._a*uni.vals._a + uni.vals._b*uni.vals._b);

    if(fabs(l) > (T)0)
    {
        uni.vals._a /= l;
        uni.vals._b /= l;
    }

	return l;
}

#endif 

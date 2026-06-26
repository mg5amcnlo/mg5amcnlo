/***
 *    ______
 *    | ___ \
 *    | |_/ /_____  __
 *    |    // _ \ \/ /
 *    | |\ \  __/>  <
 *    \_| \_\___/_/\_\
 *
 ***/
//
// *R*apid *e*vent e*x*traction Version 1.0.0
// Rex is a C++ library for parsing and manipulating Les Houches Event-format (LHE) files.
// It is designed to fast and lightweight, in comparison to internal parsers in programs like MadGraph.
// Currently, Rex is in development and may not contain all features necessary for full LHE parsing.
//
// Copyright © 2023-2026 CERN, CERN Author Zenny Wettersten.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// All rights not expressly granted are reserved.
//

#ifndef _REX_H_
#define _REX_H_

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <set>
#include <cmath>
#include <utility>
#include <memory>
#include <map>
#include <algorithm>
#include <cctype>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <queue>
#include <charconv>
#include <random>
#include <stdexcept>
#include <any>
#include <math.h>
#include <cctype>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <type_traits>
#include <cassert>
#include <variant>

// Define pi
constexpr double pi = 3.141592653589793;

// C++17 detection idiom for .size() method
template <class T, class = void>
struct has_size : std::false_type
{
};

template <class T>
struct has_size<T, std::void_t<decltype(std::declval<const T &>().size())>>
    : std::true_type
{
};

template <class T>
constexpr bool has_size_v = has_size<T>::value;

// Detects if a type is std::optional<T>
template <class T>
struct _is_optional : std::false_type
{
};
template <class U>
struct _is_optional<std::optional<U>> : std::true_type
{
};
template <class T>
constexpr bool _is_optional_v = _is_optional<T>::value;

namespace REX
{

#define UNUSED(x) (void)(x) // suppress unused variable warnings

    static const size_t npos = (size_t)-1; // generic "not found" value

    std::string to_upper(const std::string &str); // convert string to uppercase

    // generic warning function for printing warnings without throwing
    void warning(std::string message);

    // free functions / callables that take (ostream&, args...), map to string
    template <class F, class... Args,
              std::enable_if_t<!std::is_member_pointer_v<std::decay_t<F>>, int> = 0>
    std::string write_stream(F &&f, Args &&...args)
    {
        std::ostringstream ss;
        std::invoke(std::forward<F>(f), ss, std::forward<Args>(args)...);
        return ss.str();
    }

    // member functions that take (ostream&, args...) on an object, map to string
    template <class M, class Obj, class... Args,
              std::enable_if_t<std::is_member_pointer_v<std::decay_t<M>>, int> = 0>
    std::string write_stream(M mf, Obj &&obj, Args &&...args)
    {
        std::ostringstream ss;
        std::invoke(mf, std::forward<Obj>(obj), ss, std::forward<Args>(args)...);
        return ss.str();
    }

    // string trimming function to remove leading whitespace
    template <typename Str>
    Str trim_left(Str str)
    {
        auto it = str.begin();
        while (it != str.end() && std::isspace(*it))
            ++it;
        return str.substr(std::distance(str.begin(), it));
    }

    // Generic fcns for converting string-like objects to integers and doubles
    // Trims leading whitespace and strips leading '+' if present
    // Note that Str needs to have a .compare(), .data() and .size() method
    template <typename Str>
    int ctoi(Str str)
    {
        int ret;
        str = trim_left(str);
        if (str.compare(0, 1, "+") == 0)
        {
            str = str.substr(1);
        }
        auto result = std::from_chars(str.data(), str.data() + str.size(), ret);
        if (result.ec != std::errc())
        {
            throw std::invalid_argument("Invalid string-like object to convert to int");
        }
        return ret;
    }
    extern template int ctoi<std::string>(std::string str);
    extern template int ctoi<std::string_view>(std::string_view str);

    template <typename Str>
    double ctod(Str str)
    {
        double ret;
        str = trim_left(str);
        if (str.compare(0, 1, "+") == 0)
        {
            str = str.substr(1);
        }
        auto result = std::from_chars(str.data(), str.data() + str.size(), ret);
        if (result.ec != std::errc())
        {
            throw std::invalid_argument("Invalid string-like object to convert to double");
        }
        return ret;
    }
    extern template double ctod<std::string>(std::string str);
    extern template double ctod<std::string_view>(std::string_view str);

    std::string read_file(std::string_view path);
    std::vector<std::string_view> line_splitter(std::string_view content);
    std::vector<std::string_view> blank_splitter(std::string_view content);

    // ZW: index sorting function, which returns vector
    // of the indices of the original vector sorted
    // by default in ascending order
    // ie, for [5.0, 0.25, 2.0, 9.2] returns [1, 2, 0, 3]
    template <typename T>
    std::shared_ptr<std::vector<size_t>> ind_sort(const std::vector<T> &vector, std::function<bool(const T &, const T &)> comp = std::less<T>())
    {
        auto sorted = std::make_shared<std::vector<size_t>>(vector.size());
        std::iota(sorted->begin(), sorted->end(), 0);
        std::stable_sort(sorted->begin(), sorted->end(), [&](size_t i, size_t j)
                         { return comp(vector[i], vector[j]); });
        return sorted;
    }
    extern template std::shared_ptr<std::vector<size_t>> ind_sort<int>(const std::vector<int> &vector, std::function<bool(const int &, const int &)> comp = std::less<int>());
    extern template std::shared_ptr<std::vector<size_t>> ind_sort<double>(const std::vector<double> &vector, std::function<bool(const double &, const double &)> comp = std::less<double>());

    // ZW: templated fcn for multiplying two vectors elementwise,
    // assuming T has a multiplication operator*
    template <typename T>
    std::shared_ptr<std::vector<T>> vec_elem_mult(const std::vector<T> &vec1, const std::vector<T> &vec2)
    {
        if (vec1.size() < vec2.size())
        {
            return vec_elem_mult(vec2, vec1);
        }
        auto valVec = std::make_shared<std::vector<T>>(vec1.size());
        std::transform(vec1.begin(), vec1.end(), vec2.begin(), valVec->begin(), [](const T &v1, const T &v2)
                       { return v1 * v2; });
        return valVec;
    }
    extern template std::shared_ptr<std::vector<double>> vec_elem_mult(const std::vector<double> &vec1, const std::vector<double> &vec2);
    extern template std::shared_ptr<std::vector<float>> vec_elem_mult(const std::vector<float> &vec1, const std::vector<float> &vec2);
    extern template std::shared_ptr<std::vector<int>> vec_elem_mult(const std::vector<int> &vec1, const std::vector<int> &vec2);

    template <typename T>
    std::vector<T> subvector(std::vector<T> original, size_t begin, size_t end = npos)
    {
        if (end == npos)
            end = original.size();
        if (begin > end || end > original.size())
        {
            throw std::out_of_range("Invalid subvector range");
        }
        return std::vector<T>(original.begin() + begin, original.begin() + end);
    }
    extern template std::vector<int> subvector<int>(std::vector<int> original, size_t begin, size_t end);
    extern template std::vector<size_t> subvector<size_t>(std::vector<size_t> original, size_t begin, size_t end);
    extern template std::vector<short int> subvector<short int>(std::vector<short int> original, size_t begin, size_t end);
    extern template std::vector<long int> subvector<long int>(std::vector<long int> original, size_t begin, size_t end);
    extern template std::vector<double> subvector<double>(std::vector<double> original, size_t begin, size_t end);
    extern template std::vector<float> subvector<float>(std::vector<float> original, size_t begin, size_t end);
    extern template std::vector<std::string> subvector<std::string>(std::vector<std::string> original, size_t begin, size_t end);
    extern template std::vector<std::string_view> subvector<std::string_view>(std::vector<std::string_view> original, size_t begin, size_t end);

    // ================================
    // arrN<T, N, Align>
    // -----------------
    // A generic fixed-size array type,
    // used in Rex for eg 4-vectors etc
    // as STL containers do not ensure
    // contiguity of containers of containers
    // ================================

    template <typename T, size_t N, size_t Align = alignof(T)>
    struct alignas(Align) arrN
    {
        static_assert(N > 0, "N must be > 0");
        T data[N];

        constexpr arrN() = default;

        // Fill-ctor: arrN(T v) -> all elements set to v
        explicit constexpr arrN(T v)
        {
            for (size_t i = 0; i < N; ++i)
                data[i] = v;
        }

        // Element-wise ctor from N values (enforced via SFINAE)
        template <class... Args,
                  typename = std::enable_if_t<sizeof...(Args) == N &&
                                              std::conjunction<std::is_convertible<Args, T>...>::value>>
        constexpr arrN(Args &&...args) : data{T(std::forward<Args>(args))...} {}

        // From std::array
        constexpr arrN(const std::array<T, N> &a)
        {
            for (size_t i = 0; i < N; ++i)
                data[i] = a[i];
        }

        // Conversion to std::array
        constexpr operator std::array<T, N>() const
        {
            std::array<T, N> a{};
            for (size_t i = 0; i < N; ++i)
                a[i] = data[i];
            return a;
        }

        // element access
        constexpr T &operator[](size_t i)
        {
            if (i >= N)
                throw std::out_of_range("arrN[]");
            return data[i];
        }
        constexpr const T &operator[](size_t i) const
        {
            if (i >= N)
                throw std::out_of_range("arrN[]");
            return data[i];
        }
        constexpr T &operator()(size_t i) { return (*this)[i]; }
        constexpr const T &operator()(size_t i) const { return (*this)[i]; }

        // arithmetic (element-wise)
        constexpr arrN operator+(const arrN &rhs) const
        {
            arrN r;
            for (size_t i = 0; i < N; ++i)
                r.data[i] = data[i] + rhs.data[i];
            return r;
        }
        constexpr arrN operator-(const arrN &rhs) const
        {
            arrN r;
            for (size_t i = 0; i < N; ++i)
                r.data[i] = data[i] - rhs.data[i];
            return r;
        }
        constexpr arrN &operator+=(const arrN &rhs)
        {
            for (size_t i = 0; i < N; ++i)
                data[i] += rhs.data[i];
            return *this;
        }
        constexpr arrN &operator-=(const arrN &rhs)
        {
            for (size_t i = 0; i < N; ++i)
                data[i] -= rhs.data[i];
            return *this;
        }
        constexpr arrN operator*(T scalar) const
        {
            arrN r;
            for (size_t i = 0; i < N; ++i)
                r.data[i] = data[i] * scalar;
            return r;
        }
        constexpr arrN operator/(T scalar) const
        {
            arrN r;
            for (size_t i = 0; i < N; ++i)
                r.data[i] = data[i] / scalar;
            return r;
        }
        constexpr arrN &operator*=(T scalar)
        {
            for (auto &v : data)
                v *= scalar;
            return *this;
        }
        constexpr arrN &operator/=(T scalar)
        {
            for (auto &v : data)
                v /= scalar;
            return *this;
        }

        // Generic Euclidean dot
        constexpr T dot_euclidean(const arrN &rhs) const
        {
            T s{};
            for (size_t i = 0; i < N; ++i)
                s += data[i] * rhs.data[i];
            return s;
        }

        // comparisons (lexicographic)
        constexpr bool operator==(const arrN &rhs) const
        {
            for (size_t i = 0; i < N; ++i)
                if (!(data[i] == rhs.data[i]))
                    return false;
            return true;
        }
        constexpr bool operator!=(const arrN &rhs) const { return !(*this == rhs); }
        // Comparison operators --- elementwise, not generally applicable
        constexpr bool operator<(const arrN &rhs) const
        {
            for (size_t i = 0; i < N; ++i)
                if (data[i] != rhs.data[i])
                    return data[i] < rhs.data[i];
            return false;
        }
        constexpr bool operator>(const arrN &rhs) const { return rhs < *this; }
        constexpr bool operator<=(const arrN &rhs) const { return (*this < rhs) || (*this == rhs); }
        constexpr bool operator>=(const arrN &rhs) const { return (rhs < *this) || (*this == rhs); }

        // 4D-specific named accessors and Minkowski operations (enabled only when N==4)
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr T &t() { return data[0]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr const T &t() const { return data[0]; }

        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr T &x() { return data[1]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr const T &x() const { return data[1]; }

        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr T &y() { return data[2]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr const T &y() const { return data[2]; }

        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr T &z() { return data[3]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr const T &z() const { return data[3]; }

        // Minkowski dot: (+, -, -, -)
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        constexpr T dot(const arrN &other) const
        {
            return data[0] * other.data[0] - (data[1] * other.data[1] + data[2] * other.data[2] + data[3] * other.data[3]);
        }

        // Minkowski norm (sqrt(t^2 - x^2 - y^2 - z^2))
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        T norm() const
        {
            using std::sqrt;
            return sqrt(data[0] * data[0] - (data[1] * data[1] + data[2] * data[2] + data[3] * data[3]));
        }
    };

    // ================================
    // arrNRef<T, N>  (span-like proxy)
    // --------------------------------
    // A lightweight proxy for a contiguous block of memory,
    // used to treat vectors of arrN objects
    // without needing to copy data when
    // handling elements of vectors and vice versa
    // ================================
    template <typename T, size_t N>
    struct arrNRef
    {
        static_assert(N > 0, "N must be > 0");

        T *p; // points to first of N

        // ctors
        arrNRef() : p(nullptr) {}
        explicit arrNRef(T *ptr) : p(ptr) {}
        arrNRef(const arrNRef &) = default;
        arrNRef(arrNRef &&) noexcept = default;

        // element access
        T &operator[](size_t i)
        {
            assert(i < N);
            return p[i];
        }
        const T &operator[](size_t i) const
        {
            assert(i < N);
            return p[i];
        }

        // implicit read conversion to value type
        operator arrN<std::remove_const_t<T>, N>() const
        {
            arrN<std::remove_const_t<T>, N> v;
            for (size_t i = 0; i < N; ++i)
                v.data[i] = p[i];
            return v;
        }

        // assign from value
        arrNRef &operator=(const arrN<std::remove_const_t<T>, N> &v)
        {
            for (size_t i = 0; i < N; ++i)
                p[i] = v.data[i];
            return *this;
        }

        // assign from std::array
        arrNRef &operator=(const std::array<std::remove_const_t<T>, N> &a)
        {
            for (size_t i = 0; i < N; ++i)
                p[i] = a[i];
            return *this;
        }

        // assign from another proxy (copy elements)
        arrNRef &operator=(const arrNRef &rhs)
        {
            for (size_t i = 0; i < N; ++i)
                p[i] = rhs[i];
            return *this;
        }

        // assign from initializer_list
        arrNRef &operator=(std::initializer_list<std::remove_const_t<T>> ilist)
        {
            assert(ilist.size() == N);
            auto it = ilist.begin();
            for (size_t i = 0; i < N; ++i, ++it)
                p[i] = *it;
            return *this;
        }

        // comparisons (value-wise)
        bool operator==(const arrNRef &other) const
        {
            for (size_t i = 0; i < N; ++i)
                if (!(p[i] == other.p[i]))
                    return false;
            return true;
        }
        bool operator!=(const arrNRef &other) const { return !(*this == other); }

        bool operator==(const arrN<std::remove_const_t<T>, N> &other) const
        {
            for (size_t i = 0; i < N; ++i)
                if (!(p[i] == other.data[i]))
                    return false;
            return true;
        }
        bool operator!=(const arrN<std::remove_const_t<T>, N> &other) const { return !(*this == other); }

        bool operator<(const arrNRef &other) const
        {
            for (size_t i = 0; i < N; ++i)
                if (p[i] != other.p[i])
                    return p[i] < other.p[i];
            return false;
        }
        bool operator<(const arrN<std::remove_const_t<T>, N> &rhs) const
        {
            for (size_t i = 0; i < N; ++i)
                if (p[i] != rhs.data[i])
                    return p[i] < rhs.data[i];
            return false;
        }
        bool operator<=(const arrNRef &other) const { return (*this < other) || (*this == other); }
        bool operator>(const arrNRef &other) const { return other < *this; }
        bool operator>=(const arrNRef &other) const { return (other < *this) || (*this == other); }

        // 4D named accessors when N==4 (optional)
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        T &t() { return p[0]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        T &x() { return p[1]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        T &y() { return p[2]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        T &z() { return p[3]; }

        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        const T &t() const { return p[0]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        const T &x() const { return p[1]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        const T &y() const { return p[2]; }
        template <size_t M = N, typename = std::enable_if_t<M == 4>>
        const T &z() const { return p[3]; }
    };

    // ================================
    // nStrideIter<T, N>
    // -------------------------------
    // Generic iterator for vectors of
    // n-dimensional arrays arrN,
    // ie just arrN<T, N>* pointers
    // with stride N
    // ================================
    template <typename T, size_t N>
    class nStrideIter
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = arrN<T, N>;
        using difference_type = std::ptrdiff_t;
        using reference = arrNRef<T, N>;
        using pointer = void;

        nStrideIter() : ptr_(nullptr) {}
        explicit nStrideIter(T *p) : ptr_(p) {}

        reference operator*() const { return reference{ptr_}; }
        reference operator[](difference_type k) const { return reference{ptr_ + (k * static_cast<difference_type>(N))}; }

        nStrideIter &operator++()
        {
            ptr_ += N;
            return *this;
        }
        nStrideIter operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        nStrideIter &operator--()
        {
            ptr_ -= N;
            return *this;
        }
        nStrideIter operator--(int)
        {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        nStrideIter &operator+=(difference_type k)
        {
            ptr_ += k * static_cast<difference_type>(N);
            return *this;
        }
        nStrideIter &operator-=(difference_type k)
        {
            ptr_ -= k * static_cast<difference_type>(N);
            return *this;
        }

        friend nStrideIter operator+(nStrideIter it, difference_type k)
        {
            it += k;
            return it;
        }
        friend nStrideIter operator+(difference_type k, nStrideIter it)
        {
            it += k;
            return it;
        }
        friend nStrideIter operator-(nStrideIter it, difference_type k)
        {
            it -= k;
            return it;
        }
        friend difference_type operator-(nStrideIter a, nStrideIter b) { return (a.ptr_ - b.ptr_) / static_cast<difference_type>(N); }

        friend bool operator==(nStrideIter a, nStrideIter b) { return a.ptr_ == b.ptr_; }
        friend bool operator!=(nStrideIter a, nStrideIter b) { return !(a == b); }
        friend bool operator<(nStrideIter a, nStrideIter b) { return a.ptr_ < b.ptr_; }
        friend bool operator>(nStrideIter a, nStrideIter b) { return b < a; }
        friend bool operator<=(nStrideIter a, nStrideIter b) { return !(b < a); }
        friend bool operator>=(nStrideIter a, nStrideIter b) { return !(a < b); }

    private:
        T *ptr_;
    };

    // ================================
    // vecArrN<T, N>  (flat storage / N-chunk API)
    // --------------------------------
    // storage container for arrN types
    // Uses flat vector storage with
    // arrNRef access to handle internal data
    // like arrN objects
    // Uses nStride as iterator such that
    // storage traversal goes in steps of N
    // ================================

    template <typename T, size_t N>
    class vecArrN
    {
    public:
        using value_type = arrN<T, N>;
        using size_type = size_t;
        using reference = arrNRef<T, N>;
        using const_reference = arrNRef<const T, N>;
        using iterator = nStrideIter<T, N>;
        using const_iterator = nStrideIter<const T, N>;

        vecArrN() = default;
        explicit vecArrN(size_type n_chunks) : flat_(n_chunks * N) {}
        explicit vecArrN(std::vector<T> flat) : flat_(std::move(flat)) { assert(flat_.size() % N == 0); }
        template <size_type M>
        explicit vecArrN(const vecArrN<T, M> &other)
        {
            flat_ = other.flat_vector();
            size_t pad = other.size() * M % N;
            flat_.resize(flat_.size() + pad, T{});
        }

        // Construct from sequence of arrN<T, N>
        vecArrN(std::initializer_list<value_type> init)
        {
            flat_.reserve(init.size() * N);
            for (const auto &q : init)
                append_chunk(q);
        }

        // Iterator-based constructor: accepts a range of arrNRef or arrN
        template <class InputIt,
                  typename = std::enable_if_t<
                      !std::is_integral<InputIt>::value &&
                      std::is_convertible<decltype(*std::declval<InputIt>()), value_type>::value>>
        vecArrN(InputIt first, InputIt last)
        {
            flat_.reserve(std::distance(first, last) * N);
            for (; first != last; ++first)
            {
                value_type q = static_cast<value_type>(*first); // assumes arrNRef<T, N> → arrN<T, N> copy
                append_chunk(q);
            }
        }

        size_type size() const noexcept { return flat_.size() / N; }
        bool empty() const noexcept { return flat_.empty(); }

        void reserve_chunks(size_type n) { flat_.reserve(n * N); }
        size_type capacity_chunks() const noexcept { return flat_.capacity() / N; }
        void resize_chunks(size_type n) { flat_.resize(n * N); }

        reference operator[](size_type i) { return reference{flat_.data() + i * N}; }
        const_reference operator[](size_type i) const { return const_reference{flat_.data() + i * N}; }

        reference at(size_type i)
        {
            if (i >= size())
                throw std::out_of_range("vecArrN::at");
            return (*this)[i];
        }
        const_reference at(size_type i) const
        {
            if (i >= size())
                throw std::out_of_range("vecArrN::at");
            return (*this)[i];
        }

        reference front()
        {
            assert(!empty());
            return reference{flat_.data()};
        }
        const_reference front() const
        {
            assert(!empty());
            return const_reference{flat_.data()};
        }
        reference back()
        {
            assert(!empty());
            return reference{flat_.data() + (size() - 1) * N};
        }
        const_reference back() const
        {
            assert(!empty());
            return const_reference{flat_.data() + (size() - 1) * N};
        }

        iterator begin() { return iterator(flat_.data()); }
        iterator end() { return iterator(flat_.data() + flat_.size()); }
        const_iterator begin() const { return const_iterator(flat_.data()); }
        const_iterator end() const { return const_iterator(flat_.data() + flat_.size()); }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend() const { return end(); }

        void clear() noexcept { flat_.clear(); }

        void push_back(const value_type &q) { append_chunk(q); }

        template <class... Args, typename = std::enable_if_t<sizeof...(Args) == N && std::conjunction<std::is_convertible<Args, T>...>::value>>
        void push_back(Args &&...args)
        {
            value_type q{T(std::forward<Args>(args))...};
            append_chunk(q);
        }

        template <class... Args>
        void emplace_back(Args &&...args)
        {
            value_type q(std::forward<Args>(args)...);
            append_chunk(q);
        }

        iterator insert(iterator pos, const value_type &q)
        {
            auto off = pos - begin();
            auto it = flat_.insert(flat_.begin() + off * N, q.data, q.data + N);
            return iterator(flat_.data() + (it - flat_.begin()));
        }

        iterator erase(iterator pos)
        {
            auto off = pos - begin();
            auto first = flat_.begin() + off * N;
            auto it = flat_.erase(first, first + N);
            return iterator(flat_.data() + (it - flat_.begin()));
        }

        void pop_back()
        {
            assert(!empty());
            flat_.resize(flat_.size() - N);
        }

        void resize(size_type chunk_count)
        {
            flat_.resize(chunk_count * N);
        }

        void resize(size_type chunk_count, const value_type &chunk_value)
        {
            auto old_chunks = size();
            flat_.resize(chunk_count * N);
            for (size_type i = old_chunks; i < chunk_count; ++i)
            {
                (*this)[i] = chunk_value;
            }
        }

        void reserve(size_type chunk_count) { flat_.reserve(chunk_count * N); }
        size_type capacity() const noexcept { return flat_.capacity() / N; }
        void shrink_to_fit() { flat_.shrink_to_fit(); }
        size_type max_size() const noexcept { return flat_.max_size() / N; }

        void assign(size_type chunk_count, const value_type &chunk_value)
        {
            flat_.clear();
            flat_.reserve(chunk_count * N);
            for (size_type i = 0; i < chunk_count; ++i)
                append_chunk(chunk_value);
        }

        template <class InputIt>
        void assign(InputIt first, InputIt last)
        {
            flat_.clear();
            for (; first != last; ++first)
            {
                value_type q = static_cast<value_type>(*first);
                append_chunk(q);
            }
        }

        iterator insert(iterator pos, size_type count, const value_type &chunk_value)
        {
            auto off = pos - begin();
            std::vector<T> tmp;
            tmp.reserve(count * N);
            for (size_type i = 0; i < count; ++i)
            {
                tmp.insert(tmp.end(), std::begin(chunk_value.data), std::end(chunk_value.data));
            }
            auto it = flat_.insert(flat_.begin() + off * N, tmp.begin(), tmp.end());
            return iterator(flat_.data() + (it - flat_.begin()));
        }

        template <class InputIt>
        iterator insert(iterator pos, InputIt first, InputIt last)
        {
            auto off = pos - begin();
            std::vector<T> tmp;
            for (; first != last; ++first)
            {
                value_type q = static_cast<value_type>(*first);
                tmp.insert(tmp.end(), std::begin(q.data), std::end(q.data));
            }
            auto it = flat_.insert(flat_.begin() + off * N, tmp.begin(), tmp.end());
            return iterator(flat_.data() + (it - flat_.begin()));
        }

        iterator erase(iterator first, iterator last)
        {
            auto off1 = first - begin();
            auto off2 = last - begin();
            auto it = flat_.erase(flat_.begin() + off1 * N, flat_.begin() + off2 * N);
            return iterator(flat_.data() + (it - flat_.begin()));
        }

        void swap(vecArrN &other) noexcept(noexcept(flat_.swap(other.flat_)))
        {
            flat_.swap(other.flat_);
        }

        bool operator==(const vecArrN &other) const noexcept { return flat_ == other.flat_; }
        bool operator!=(const vecArrN &other) const noexcept { return !(*this == other); }

        std::vector<T> &flat_vector() noexcept { return flat_; }
        const std::vector<T> &flat_vector() const noexcept { return flat_; }

        // append from raw memory (unsafe unless count%N==0)
        void append_flat(const T *data, size_type count)
        {
            assert(count % N == 0);
            flat_.insert(flat_.end(), data, data + count);
        }

        vecArrN subvec(size_type start_chunk, size_type end_chunk) const
        {
            if (start_chunk > end_chunk || end_chunk > size())
            {
                throw std::out_of_range("vecArrN::subvec - invalid range");
            }
            return vecArrN(begin() + start_chunk, begin() + end_chunk);
        }

        template <size_t M>
        vecArrN<T, M> transpose() const
        {
            static_assert(M == this->size(), "Invalid transpose size");
            vecArrN<T, M> result(N);
            for (size_type i = 0; i < N; ++i)
            {
                for (size_type j = 0; j < M; ++j)
                {
                    result[j][i] = (*this)[i][j];
                }
            }
            return result;
        }

    private:
        void append_chunk(const value_type &q)
        {
            flat_.insert(flat_.end(), std::begin(q.data), std::end(q.data));
        }

        std::vector<T> flat_;
    };

    template <typename T, size_t N>
    vecArrN<T, N> subvector(vecArrN<T, N> original, size_t begin, size_t end = npos)
    {
        if (end == npos)
            end = original.size();
        if (begin > end || end > original.size())
        {
            throw std::out_of_range("Invalid subvector range");
        }
        return original.subvec(begin, end);
    }

    // Defining specific instances for arr2, arr3, and arr4
    template <typename T>
    using arr2 = arrN<T, 2>;
    extern template struct arrN<short int, 2>;
    extern template struct arrN<long int, 2>;
    extern template struct arrN<int, 2>;
    extern template struct arrN<float, 2>;
    extern template struct arrN<double, 2>;
    template <typename T>
    using arr3 = arrN<T, 3>;
    extern template struct arrN<short int, 3>;
    extern template struct arrN<long int, 3>;
    extern template struct arrN<int, 3>;
    extern template struct arrN<float, 3>;
    extern template struct arrN<double, 3>;
    template <typename T>
    using arr4 = arrN<T, 4>;
    extern template struct arrN<short int, 4>;
    extern template struct arrN<long int, 4>;
    extern template struct arrN<int, 4>;
    extern template struct arrN<float, 4>;
    extern template struct arrN<double, 4>;

    template <typename T>
    using arr2Ref = arrNRef<T, 2>;
    extern template struct arrNRef<short int, 2>;
    extern template struct arrNRef<long int, 2>;
    extern template struct arrNRef<int, 2>;
    extern template struct arrNRef<float, 2>;
    extern template struct arrNRef<double, 2>;
    template <typename T>
    using arr3Ref = arrNRef<T, 3>;
    extern template struct arrNRef<short int, 3>;
    extern template struct arrNRef<long int, 3>;
    extern template struct arrNRef<int, 3>;
    extern template struct arrNRef<float, 3>;
    extern template struct arrNRef<double, 3>;
    template <typename T>
    using arr4Ref = arrNRef<T, 4>;
    extern template struct arrNRef<short int, 4>;
    extern template struct arrNRef<long int, 4>;
    extern template struct arrNRef<int, 4>;
    extern template struct arrNRef<float, 4>;
    extern template struct arrNRef<double, 4>;

    template <typename T>
    using vecArr2 = vecArrN<T, 2>;
    extern template struct vecArrN<short int, 2>;
    extern template struct vecArrN<long int, 2>;
    extern template struct vecArrN<int, 2>;
    extern template struct vecArrN<float, 2>;
    extern template struct vecArrN<double, 2>;
    template <typename T>
    using vecArr3 = vecArrN<T, 3>;
    extern template struct vecArrN<short int, 3>;
    extern template struct vecArrN<long int, 3>;
    extern template struct vecArrN<int, 3>;
    extern template struct vecArrN<float, 3>;
    extern template struct vecArrN<double, 3>;
    template <typename T>
    using vecArr4 = vecArrN<T, 4>;
    extern template struct vecArrN<short int, 4>;
    extern template struct vecArrN<long int, 4>;
    extern template struct vecArrN<int, 4>;
    extern template struct vecArrN<float, 4>;
    extern template struct vecArrN<double, 4>;

    // Explicit parton struct for handling particle objects
    // outside of the view-like type stored in event objects
    struct parton
    {
        arr4<double> momenta_ = {0.0, 0.0, 0.0, 0.0}; // (E, px, py, pz)
        double mass_ = 0.0;                           // mass
        double vtim_ = 0.0;                           // lifetime
        double spin_ = 0.0;                           // spin
        long int pdg_ = 0;                            // PDG ID
        short int status_ = 0;                        // status
        arr2<short int> mother_ = {0, 0};             // mother IDs
        arr2<short int> icol_ = {0, 0};               // color IDs

        parton() = default;
        parton(const arr4<double> &mom, double m, double v,
               double s, long int p, short int st,
               arr2<short int> &mth, arr2<short int> &col)
            : momenta_(mom), mass_(m), vtim_(v), spin_(s), pdg_(p), status_(st), mother_(mth), icol_(col) {}
        parton(arr4<double> &&mom, double m, double v,
               double s, long int p, short int st,
               arr2<short int> &&mth, arr2<short int> &&col)
            : momenta_(std::move(mom)), mass_(m), vtim_(v), spin_(s), pdg_(p), status_(st), mother_(std::move(mth)), icol_(std::move(col)) {}
        parton(const parton &) = default;
        parton(parton &&) noexcept = default;
        parton &operator=(const parton &) = default;
        parton &operator=(parton &&) noexcept = default;

        // Getters
        arr4<double> &momenta();
        const arr4<double> &momenta() const;
        arr4<double> &momentum();
        const arr4<double> &momentum() const;
        arr4<double> &pUP();
        const arr4<double> &pUP() const;
        arr4<double> &p();
        const arr4<double> &p() const;
        arr4<double> &mom();
        const arr4<double> &mom() const;
        double &E();
        const double &E() const;
        double &t();
        const double &t() const;
        double &px();
        const double &px() const;
        double &x();
        const double &x() const;
        double &py();
        const double &py() const;
        double &y();
        const double &y() const;
        double &pz();
        const double &pz() const;
        double &z();
        const double &z() const;
        double &m();
        const double &m() const;
        double &mass();
        const double &mass() const;
        double &vtim();
        const double &vtim() const;
        double &vTimUP();
        const double &vTimUP() const;
        double &spin();
        const double &spin() const;
        double &spinUP();
        const double &spinUP() const;
        long int &pdg();
        const long int &pdg() const;
        long int &idUP();
        const long int &idUP() const;
        long int &id();
        const long int &id() const;
        short int &status();
        const short int &status() const;
        short int &iStUP();
        const short int &iStUP() const;
        short int &iSt();
        const short int &iSt() const;
        arr2<short int> &mother();
        const arr2<short int> &mother() const;
        arr2<short int> &mothUP();
        const arr2<short int> &mothUP() const;
        arr2<short int> &moth();
        const arr2<short int> &moth() const;
        arr2<short int> &icol();
        const arr2<short int> &icol() const;
        arr2<short int> &iColUP();
        const arr2<short int> &iColUP() const;
        arr2<short int> &iCol();
        const arr2<short int> &iCol() const;

        // Self-returning setters
        parton &set_momenta(const arr4<double> &mom);
        parton &set_pUP(const arr4<double> &mom);
        parton &set_p(const arr4<double> &mom);
        parton &set_mom(const arr4<double> &mom);
        parton &set_E(double E);
        parton &set_t(double pt);
        parton &set_px(double px);
        parton &set_x(double x);
        parton &set_py(double py);
        parton &set_y(double y);
        parton &set_pz(double pz);
        parton &set_z(double z);
        parton &set_mass(double m);
        parton &set_vtim(double v);
        parton &set_vTimUP(double v);
        parton &set_spin(double s);
        parton &set_spinUP(double s);
        parton &set_pdg(long int p);
        parton &set_idUP(long int p);
        parton &set_id(long int p);
        parton &set_status(short int st);
        parton &set_iStUP(short int st);
        parton &set_iSt(short int st);
        parton &set_mother(const arr2<short int> &mth);
        parton &set_mothUP(const arr2<short int> &mth);
        parton &set_moth(const arr2<short int> &mth);
        parton &set_mother(const short int m1, const short int m2);
        parton &set_mothUP(const short int m1, const short int m2);
        parton &set_moth(const short int m1, const short int m2);
        parton &set_icol(const arr2<short int> &col);
        parton &set_iColUP(const arr2<short int> &col);
        parton &set_icol(const short int c1, const short int c2);
        parton &set_iColUP(const short int c1, const short int c2);
        parton &set_iCol(const arr2<short int> &col);
        parton &set_iCol(const short int c1, const short int c2);

        // Calculated observables
        double pT() const;    // transverse momentum
        double pT2() const;   // transverse momentum squared
        double pL() const;    // longitudinal momentum
        double pL2() const;   // longitudinal momentum squared
        double eT() const;    // transverse energy
        double eT2() const;   // transverse energy squared
        double phi() const;   // azimuthal angle
        double theta() const; // polar angle
        double eta() const;   // pseudorapidity
        double rap() const;   // rapidity
        double mT() const;    // transverse mass
        double mT2() const;   // transverse mass squared
        double m2() const;    // mass squared
    };

    struct event
    {
    public:
        // Default constructors
        event() = default;
        event(const event &) = default;
        event(event &&) noexcept = default;
        event &operator=(const event &) = default;
        event &operator=(event &&) noexcept = default;
        // Constructor with number of particles
        explicit event(size_t n_particles);
        explicit event(std::vector<parton> particles);

        size_t n_ = 0;                                          // number of partons in the event
        long int proc_id_ = 0;                                  // process ID
        double weight_ = 0.0;                                   // event weight
        double scale_ = 0.0;                                    // event scale
        double muF_ = 0.0;                                      // factorization scale
        double muR_ = 0.0;                                      // renormalization scale
        double muPS_ = 0.0;                                     // parton shower scale
        double alphaEW_ = 0.0;                                  // electromagnetic coupling constant
        double alphaS_ = 0.0;                                   // strong coupling constant
        vecArr4<double> momenta_ = {};                          // momenta of particles (E, px, py, pz)
        std::vector<double> mass_ = {}, vtim_ = {}, spin_ = {}; // mass, virtual time, and spin
        std::vector<long int> pdg_ = {};                        // particle ids according to PDG standard
        std::vector<short int> status_ = {};                    // particle statuses in LHE standard (ie -1 incoming, +1 outgoing etc)
        vecArr2<short int> mother_ = {}, icol_ = {};            // mother and color indices
        std::vector<double> wgts_ = {};                         // additional weights, if any; note that wgt ids are not stored at the event level, so custom writers need to handle this at the LHEF level

        // Self-returning setters
        event &set_n(size_t n);
        event &set_proc_id(long int id);
        event &set_weight(double w);
        event &set_scale(double s);
        event &set_muF(double muF);
        event &set_muR(double muR);
        event &set_muPS(double muPS);
        event &set_alphaEW(double aew);
        event &set_alphaS(double as);
        event &set_momenta(const vecArr4<double> &mom);
        event &set_momenta(const std::vector<std::array<double, 4>> &mom);
        event &set_mass(const std::vector<double> &m);
        event &set_vtim(const std::vector<double> &v);
        event &set_spin(const std::vector<double> &s);
        event &set_pdg(const std::vector<long int> &p);
        event &set_status(const std::vector<short int> &st);
        event &set_mother(const vecArr2<short int> &m);
        event &set_mother(const std::vector<std::array<short int, 2>> &m);
        event &set_icol(const vecArr2<short int> &c);
        event &set_icol(const std::vector<std::array<short int, 2>> &c);
        event &set_wgts(const std::vector<double> &w);
        event &add_wgt(double w, const std::string &id = "");

        std::vector<size_t> indices = {};                                  // indices of particles for ordered views without modifying underlying data
        event &set_indices();                                              // Default indexing is sequential by storage order
        event &set_indices(const event &e, bool fail_on_mismatch = false); // Set indices based on another event (fail on mismatch forces events to be equal, throws on miss)
        event &set_indices(const std::vector<size_t> &idxs);               // Set indices explicitly

        // Access functions for alternative names of variables
        size_t &nUP();
        const size_t &nUP() const;
        size_t &n();
        const size_t &n() const;
        long int &idPrUP();
        const long int &idPrUP() const;
        long int &idPr();
        const long int &idPr() const;
        double &xWgtUP();
        const double &xWgtUP() const;
        double &xWgt();
        const double &xWgt() const;
        double &weight();
        const double &weight() const;
        double &scale();
        const double &scale() const;
        double &scalUP();
        const double &scalUP() const;
        double &muF();
        const double &muF() const;
        double &muR();
        const double &muR() const;
        double &muPS();
        const double &muPS() const;
        double &aQEDUP();
        const double &aQEDUP() const;
        double &alphaQED();
        const double &alphaQED() const;
        double &aQED();
        const double &aQED() const;
        double &alphaEW();
        const double &alphaEW() const;
        double &aEW();
        const double &aEW() const;
        double &aQCDUP();
        const double &aQCDUP() const;
        double &alphaS();
        const double &alphaS() const;
        double &aS();
        const double &aS() const;
        double &aQCD();
        const double &aQCD() const;
        vecArr4<double> &momenta();
        const vecArr4<double> &momenta() const;
        vecArr4<double> &momentum();
        const vecArr4<double> &momentum() const;
        vecArr4<double> &pUP();
        const vecArr4<double> &pUP() const;
        vecArr4<double> &p();
        const vecArr4<double> &p() const;
        std::vector<double> &mUP();
        const std::vector<double> &mUP() const;
        std::vector<double> &m();
        const std::vector<double> &m() const;
        std::vector<double> &mass();
        const std::vector<double> &mass() const;
        std::vector<double> &vtim();
        const std::vector<double> &vtim() const;
        std::vector<double> &vTimUP();
        const std::vector<double> &vTimUP() const;
        std::vector<double> &vTim();
        const std::vector<double> &vTim() const;
        std::vector<double> &spin();
        const std::vector<double> &spin() const;
        std::vector<double> &spinUP();
        const std::vector<double> &spinUP() const;
        std::vector<long int> &idUP();
        const std::vector<long int> &idUP() const;
        std::vector<long int> &id();
        const std::vector<long int> &id() const;
        std::vector<long int> &pdg();
        const std::vector<long int> &pdg() const;
        std::vector<short int> &iStUP();
        const std::vector<short int> &iStUP() const;
        std::vector<short int> &status();
        const std::vector<short int> &status() const;
        std::vector<short int> &iSt();
        const std::vector<short int> &iSt() const;
        vecArr2<short int> &mother();
        const vecArr2<short int> &mother() const;
        vecArr2<short int> &mothUP();
        const vecArr2<short int> &mothUP() const;
        vecArr2<short int> &moth();
        const vecArr2<short int> &moth() const;
        vecArr2<short int> &icol();
        const vecArr2<short int> &icol() const;
        vecArr2<short int> &iColUP();
        const vecArr2<short int> &iColUP() const;
        vecArr2<short int> &iCol();
        const vecArr2<short int> &iCol() const;
        std::vector<double> &wgts();
        const std::vector<double> &wgts() const;
        size_t n_wgts() const;

        // IDs for various additional weights, shared between events (and the LHE struct)
        std::shared_ptr<std::vector<std::string>> weight_ids = nullptr;

        // Print functions (LHEF XML format)
        void print_head(std::ostream &os = std::cout) const;
        void print_wgts_ids(std::ostream &os = std::cout) const;
        void print_wgts_no_ids(std::ostream &os = std::cout) const;
        void print_wgts(std::ostream &os = std::cout, bool include_ids = false) const;
        void print(std::ostream &os = std::cout, bool include_ids = false) const;
        void print_extra(std::ostream &os = std::cout) const;
        void print_scales(std::ostream &os = std::cout) const;

        // Calculates gS based on alphaS
        double gS();

        // Scales (returns LHE scale if not set)
        double get_muF() const;
        double get_muR() const;
        double get_muPS() const;

        // Particle struct, gives a view of the corresponding
        // elements of the event-level storage vectors
        struct particle
        {
            // Use arrNRef for reference-like access to arrN-like objects
            // despite being sequences of elements of vecArrN objects
            arr4Ref<double> momentum_;
            double &mass_;
            double &vtim_;
            double &spin_;
            long int &pdg_;
            short int &status_;
            arr2Ref<short int> mother_;
            arr2Ref<short int> icol_;

            particle(arr4Ref<double> mom,
                     double &m, double &v, double &s,
                     long int &p, short int &st,
                     arr2Ref<short int> mth,
                     arr2Ref<short int> col)
                : momentum_(mom), mass_(m), vtim_(v), spin_(s),
                  pdg_(p), status_(st), mother_(mth), icol_(col) {}

            particle(arr4Ref<const double> mom,
                     const double &m, const double &v, const double &s,
                     const long int &p, const short int &st,
                     arr2Ref<const short int> mth,
                     arr2Ref<const short int> col)
                : momentum_(arr4Ref<double>{const_cast<double *>(mom.p)}),
                  mass_(const_cast<double &>(m)), vtim_(const_cast<double &>(v)), spin_(const_cast<double &>(s)),
                  pdg_(const_cast<long int &>(p)), status_(const_cast<short int &>(st)),
                  mother_(arr2Ref<short int>{const_cast<short int *>(mth.p)}),
                  icol_(arr2Ref<short int>{const_cast<short int *>(col.p)}) {}

            arr4Ref<double> pUP();
            arr4Ref<const double> pUP() const;
            arr4Ref<double> mom();
            arr4Ref<const double> mom() const;
            arr4Ref<double> p();
            arr4Ref<const double> p() const;
            arr4Ref<double> momentum();
            arr4Ref<const double> momentum() const;

            // Component aliases
            double &E();
            const double &E() const;
            double &t();
            const double &t() const;
            double &x();
            double &px();
            const double &x() const;
            const double &px() const;
            double &y();
            double &py();
            const double &y() const;
            const double &py() const;
            double &z();
            double &pz();
            const double &z() const;
            const double &pz() const;
            double &mUP();
            const double &mUP() const;
            double &m();
            const double &m() const;
            double &mass();
            const double &mass() const;
            double &vtim();
            const double &vtim() const;
            double &vTimUP();
            const double &vTimUP() const;
            double &vTim();
            const double &vTim() const;
            double &spin();
            const double &spin() const;
            double &spinUP();
            const double &spinUP() const;
            long int &idUP();
            const long int &idUP() const;
            long int &id();
            const long int &id() const;
            long int &pdg();
            const long int &pdg() const;
            short int &status();
            const short int &status() const;
            short int &iSt();
            const short int &iSt() const;
            short int &iStUP();
            const short int &iStUP() const;
            arr2Ref<short int> mothUP();
            const arr2Ref<short int> mothUP() const;
            arr2Ref<short int> moth();
            const arr2Ref<short int> moth() const;
            arr2Ref<short int> mother();
            const arr2Ref<short int> mother() const;
            arr2Ref<short int> icol();
            const arr2Ref<short int> icol() const;
            arr2Ref<short int> iColUP();
            const arr2Ref<short int> iColUP() const;
            arr2Ref<short int> iCol();
            const arr2Ref<short int> iCol() const;

            particle &set_pdg(long int p);
            particle &set_id(long int p);
            particle &set_idUP(long int p);
            particle &set_status(short int s);
            particle &set_iSt(short int s);
            particle &set_iStUP(short int s);
            particle &set_mother(short int i, short int j);
            particle &set_mother(const arr2<short int> &m);
            particle &set_moth(short int i, short int j);
            particle &set_moth(const arr2<short int> &m);
            particle &set_mothUP(short int i, short int j);
            particle &set_mothUP(const arr2<short int> &m);
            particle &set_icol(short int i, short int c);
            particle &set_icol(const arr2<short int> &c);
            particle &set_iColUP(short int i, short int c);
            particle &set_iColUP(const arr2<short int> &c);
            particle &set_iCol(short int i, short int c);
            particle &set_iCol(const arr2<short int> &c);
            particle &set_momentum(double e, double px, double py, double pz);
            particle &set_momentum(const arr4<double> &mom);
            particle &set_mom(double e, double px, double py, double pz);
            particle &set_mom(const arr4<double> &mom);
            particle &set_pUP(double e, double px, double py, double pz);
            particle &set_pUP(const arr4<double> &mom);
            particle &set_p(double e, double px, double py, double pz);
            particle &set_p(const arr4<double> &mom);
            particle &set_E(double e);
            particle &set_t(double pt);
            particle &set_x(double x);
            particle &set_px(double px);
            particle &set_y(double y);
            particle &set_py(double py);
            particle &set_z(double z);
            particle &set_pz(double pz);
            particle &set_mass(double m);
            particle &set_mUP(double m);
            particle &set_m(double m);
            particle &set_vtim(double v);
            particle &set_vTimUP(double v);
            particle &set_vTim(double v);
            particle &set_spin(double s);
            particle &set_spinUP(double s);

            // Calculated observables
            double pT() const;    // transverse momentum
            double pT2() const;   // transverse momentum squared
            double pL() const;    // longitudinal momentum
            double pL2() const;   // longitudinal momentum squared
            double eT() const;    // transverse energy
            double eT2() const;   // transverse energy squared
            double phi() const;   // azimuthal angle
            double theta() const; // polar angle
            double eta() const;   // pseudorapidity
            double rap() const;   // rapidity
            double mT() const;    // transverse mass
            double mT2() const;   // transverse mass squared
            double m2() const;    // mass squared

            // Print particle information (LHEF XML format)
            void print(std::ostream &os = std::cout) const;
        };

        // Const version of the particle struct
        struct const_particle
        {
            arr4Ref<const double> momentum_;
            const double &mass_;
            const double &vtim_;
            const double &spin_;
            const long int &pdg_;
            const short int &status_;
            arr2Ref<const short int> mother_;
            arr2Ref<const short int> icol_;

            const_particle(arr4Ref<const double> mom,
                           const double &m, const double &v, const double &s,
                           const long int &p, const short int &st,
                           arr2Ref<const short int> mth,
                           arr2Ref<const short int> col)
                : momentum_(mom), mass_(m), vtim_(v), spin_(s),
                  pdg_(p), status_(st), mother_(mth), icol_(col) {}

            arr4Ref<const double> pUP() const;
            arr4Ref<const double> mom() const;
            arr4Ref<const double> p() const;
            arr4Ref<const double> momentum() const;
            // Component aliases
            const double &E() const;
            const double &t() const;
            const double &x() const;
            const double &px() const;
            const double &y() const;
            const double &py() const;
            const double &z() const;
            const double &pz() const;
            const double &mUP() const;
            const double &m() const;
            const double &mass() const;
            const double &vtim() const;
            const double &vTimUP() const;
            const double &vTim() const;
            const double &spin() const;
            const double &spinUP() const;
            const long int &idUP() const;
            const long int &id() const;
            const long int &pdg() const;
            const short int &status() const;
            const short int &iSt() const;
            const short int &iStUP() const;
            arr2Ref<const short int> mothUP() const;
            arr2Ref<const short int> moth() const;
            arr2Ref<const short int> mother() const;
            arr2Ref<const short int> icol() const;
            arr2Ref<const short int> iColUP() const;
            arr2Ref<const short int> iCol() const;

            // Calculated observables
            double pT() const;    // transverse momentum
            double pT2() const;   // transverse momentum squared
            double pL() const;    // longitudinal momentum
            double pL2() const;   // longitudinal momentum squared
            double eT() const;    // transverse energy
            double eT2() const;   // transverse energy squared
            double phi() const;   // azimuthal angle
            double theta() const; // polar angle
            double eta() const;   // pseudorapidity
            double rap() const;   // rapidity
            double mT() const;    // transverse mass
            double mT2() const;   // transverse mass squared
            double m2() const;    // mass squared

            void print(std::ostream &os = std::cout) const;
        };

        // Access particle views by indexing event like a vector
        particle operator[](size_t i);
        particle at(size_t i);
        const_particle operator[](size_t i) const;
        const_particle at(size_t i) const;
        particle get_particle(size_t i);
        const_particle get_particle(size_t i) const;
        size_t size() const;

        struct particle_iterator
        {
            event *evt;
            size_t index;
            particle operator*();
            particle_iterator &operator++();
            bool operator!=(const particle_iterator &) const;
        };

        struct const_particle_iterator
        {
            const event *evt;
            size_t index;
            const_particle operator*() const;
            const_particle_iterator &operator++();
            bool operator!=(const const_particle_iterator &) const;
        };

        particle_iterator begin();
        particle_iterator end();
        const_particle_iterator begin() const;
        const_particle_iterator end() const;

        // Add particle to event, sets n = n+1
        event &add_particle(const parton &p);
        event &add_particle(const particle &p);
        event &add_particle(const const_particle &p);

        // Throws if any vector has a mismatched size with each other or nUP
        void validate() const;

        // Generic additional tags/information
        std::unordered_map<std::string, std::any> extra;

        // Set generic data (overwrites if exists)
        // Note: std::any is used to allow any type of value to be stored
        // This is a simple key-value store for data, not part of the LHEF standard
        template <typename T>
        void set(const std::string &name, T value)
        {
            extra[name] = std::any(std::move(value));
        }

        // Get extra data (throws if not found or wrong type)
        template <typename T>
        T &get(const std::string &name)
        {
            auto it = extra.find(name);
            if (it == extra.end())
            {
                throw std::out_of_range("event::get: No parameter named '" + name + "'");
            }
            if (it->second.type() != typeid(T))
            {
                throw std::runtime_error("event::get: Parameter '" + name + "' is not of requested type");
            }
            return std::any_cast<T &>(it->second);
        }

        template <typename T>
        const T &get(const std::string &name) const
        {
            auto it = extra.find(name);
            if (it == extra.end())
            {
                throw std::out_of_range("event::get: No parameter named '" + name + "'");
            }
            if (it->second.type() != typeid(T))
            {
                throw std::bad_any_cast("event::get: Parameter '" + name + "' is not of requested type");
            }
            return std::any_cast<const T &>(it->second);
        }

        bool has(const std::string &name) const
        {
            return extra.find(name) != extra.end();
        }

        // Internal view for event to iterate over particles as indices,
        // ie allows treating the event as a collection of particles
        // according to the indices vector ordering
        struct event_view
        {
            event &evt;
            const std::vector<size_t> &indices;

            struct iterator
            {
                event &evt;
                const std::vector<size_t> &indices;
                size_t i;

                iterator(event &evt_, const std::vector<size_t> &indices_, size_t idx)
                    : evt(evt_), indices(indices_), i(idx) {}

                event::particle operator*()
                {
                    return evt.get_particle(indices[i]);
                }

                iterator &operator++()
                {
                    ++i;
                    return *this;
                }

                bool operator!=(const iterator &other) const
                {
                    return i != other.i || &indices != &other.indices;
                }
            };

            iterator begin() { return iterator{evt, indices, 0}; }
            iterator end() { return iterator{evt, indices, indices.size()}; }

            size_t size() const { return indices.size(); }

            event::particle operator[](size_t i)
            {
                return evt.get_particle(indices[i]);
            }
        };

        struct const_event_view
        {
            const event &evt;
            const std::vector<size_t> &indices;

            struct iterator
            {
                const event &evt;
                const std::vector<size_t> &indices;
                size_t i;

                iterator(const event &evt_, const std::vector<size_t> &indices_, size_t idx)
                    : evt(evt_), indices(indices_), i(idx) {}

                event::const_particle operator*() const
                {
                    return evt.get_particle(indices[i]);
                }

                iterator &operator++()
                {
                    ++i;
                    return *this;
                }

                bool operator!=(const iterator &other) const
                {
                    return i != other.i || &indices != &other.indices;
                }
            };

            iterator begin() const { return iterator{evt, indices, 0}; }
            iterator end() const { return iterator{evt, indices, indices.size()}; }

            size_t size() const { return indices.size(); }

            event::const_particle operator[](size_t i) const
            {
                return evt.get_particle(indices[i]);
            }
        };

        event_view view()
        {
            if (this->indices.empty())
                this->set_indices();
            return event_view{*this, indices};
        }

        const_event_view view() const
        {
            if (this->indices.empty())
                const_cast<event *>(this)->set_indices();
            return const_event_view{*this, indices};
        }
    };

    // Event comparator type
    using event_equal_fn = std::function<bool(event &, event &)>;
    using cevent_equal_fn = std::function<bool(const event &, const event &)>;

    // Global access
    bool default_event_equal(const event &lhs, const event &rhs);

    // Custom comparator interface
    bool operator==(const event &lhs, const event &rhs);
    bool operator!=(const event &lhs, const event &rhs);
    void set_event_comparator(cevent_equal_fn fn);
    void reset_event_comparator();
    bool external_legs_comparator(event &a, event &b);
    bool external_legs_const_comparator(const event &a, const event &b);
    bool always_true(const event &a, const event &b);

    // Class to create custom event comparators
    // This class allows users to define custom comparison logic for events
    // by specifying which fields to compare and their tolerances.
    // It supports comparison for all standard LHEF fields, and will automatically
    // sort particle-level fields using exclusively the fields specified in the configuration.
    // Additionally, the status_filter variable allows for defining which particle statuses to extract for comparison.
    // For doubles, relative tolerances can be set independently for each field by the user.
    // However, for integers, only exact equality is supported.
    struct eventComparatorConfig
    {
        // Status filter: only compare particles with one of these statuses
        std::set<int> status_filter = {}; // empty = no filtering

        bool compare_momentum = false;
        // per-momentum component toggles
        bool compare_momentum_x = false;
        bool compare_momentum_y = false;
        bool compare_momentum_z = false;
        bool compare_momentum_E = false;

        bool compare_mass = true;
        bool compare_vtim = false;
        bool compare_spin = false;
        bool compare_pdg = true;
        bool compare_status = true;
        bool compare_mother = false;
        bool compare_icol = false;

        bool compare_n = false;
        bool compare_proc_id = false;
        bool compare_weight = false;
        bool compare_scale = false;
        bool compare_alphaEW = false;
        bool compare_alphaS = false;

        double mass_tol = 1e-8;
        double vtim_tol = 1e-8;
        double spin_tol = 1e-8;
        double momentum_tol = 1e-8;
        double weight_tol = 1e-8;
        double scale_tol = 1e-8;
        double alphaEW_tol = 1e-8;
        double alphaS_tol = 1e-8;

        event_equal_fn make_comparator() const;
        cevent_equal_fn make_const_comparator() const;
        // Convenience functions to set individual parameters
        // Event-level parameters
        eventComparatorConfig &set_n(bool v)
        {
            compare_n = v;
            return *this;
        }
        eventComparatorConfig &set_nUP(bool v)
        {
            compare_n = v;
            return *this;
        }
        eventComparatorConfig &set_proc_id(bool v)
        {
            compare_proc_id = v;
            return *this;
        }
        eventComparatorConfig &set_idPr(bool v)
        {
            compare_proc_id = v;
            return *this;
        }
        eventComparatorConfig &set_idPrUP(bool v)
        {
            compare_proc_id = v;
            return *this;
        }
        eventComparatorConfig &set_weight(bool v)
        {
            compare_weight = v;
            return *this;
        }
        eventComparatorConfig &set_xWgt(bool v)
        {
            compare_weight = v;
            return *this;
        }
        eventComparatorConfig &set_xWgtUP(bool v)
        {
            compare_weight = v;
            return *this;
        }
        eventComparatorConfig &set_scale(bool v)
        {
            compare_scale = v;
            return *this;
        }
        eventComparatorConfig &set_scalUP(bool v)
        {
            compare_scale = v;
            return *this;
        }
        eventComparatorConfig &set_alphaEW(bool v)
        {
            compare_alphaEW = v;
            return *this;
        }
        eventComparatorConfig &set_aQED(bool v)
        {
            compare_alphaEW = v;
            return *this;
        }
        eventComparatorConfig &set_aQEDUP(bool v)
        {
            compare_alphaEW = v;
            return *this;
        }
        eventComparatorConfig &set_alphaQED(bool v)
        {
            compare_alphaEW = v;
            return *this;
        }
        eventComparatorConfig &set_aEW(bool v)
        {
            compare_alphaEW = v;
            return *this;
        }
        eventComparatorConfig &set_alphaS(bool v)
        {
            compare_alphaS = v;
            return *this;
        }
        eventComparatorConfig &set_aQCD(bool v)
        {
            compare_alphaS = v;
            return *this;
        }
        eventComparatorConfig &set_aQCDUP(bool v)
        {
            compare_alphaS = v;
            return *this;
        }
        eventComparatorConfig &set_aS(bool v)
        {
            compare_alphaS = v;
            return *this;
        }
        // Particle-specific parameters
        eventComparatorConfig &set_momentum(bool v)
        {
            compare_momentum = v;
            compare_momentum_x = v;
            compare_momentum_y = v;
            compare_momentum_z = v;
            compare_momentum_E = v;
            return *this;
        }
        eventComparatorConfig &set_pUP(bool v)
        {
            compare_momentum = v;
            compare_momentum_x = v;
            compare_momentum_y = v;
            compare_momentum_z = v;
            compare_momentum_E = v;
            return *this;
        }
        eventComparatorConfig &set_p(bool v)
        {
            compare_momentum = v;
            compare_momentum_x = v;
            compare_momentum_y = v;
            compare_momentum_z = v;
            compare_momentum_E = v;
            return *this;
        }
        eventComparatorConfig &set_momenta(bool v)
        {
            compare_momentum = v;
            compare_momentum_x = v;
            compare_momentum_y = v;
            compare_momentum_z = v;
            compare_momentum_E = v;
            return *this;
        }
        eventComparatorConfig &set_mom(bool v)
        {
            compare_momentum = v;
            compare_momentum_x = v;
            compare_momentum_y = v;
            compare_momentum_z = v;
            compare_momentum_E = v;
            return *this;
        }
        eventComparatorConfig &set_momentum(bool e, bool x, bool y, bool z)
        {
            compare_momentum = e || x || y || z;
            compare_momentum_E = e;
            compare_momentum_x = x;
            compare_momentum_y = y;
            compare_momentum_z = z;
            return *this;
        }
        eventComparatorConfig &set_momenta(bool e, bool x, bool y, bool z)
        {
            compare_momentum = e || x || y || z;
            compare_momentum_E = e;
            compare_momentum_x = x;
            compare_momentum_y = y;
            compare_momentum_z = z;
            return *this;
        }
        eventComparatorConfig &set_pUP(bool e, bool x, bool y, bool z)
        {
            compare_momentum = e || x || y || z;
            compare_momentum_E = e;
            compare_momentum_x = x;
            compare_momentum_y = y;
            compare_momentum_z = z;
            return *this;
        }
        eventComparatorConfig &set_p(bool e, bool x, bool y, bool z)
        {
            compare_momentum = e || x || y || z;
            compare_momentum_E = e;
            compare_momentum_x = x;
            compare_momentum_y = y;
            compare_momentum_z = z;
            return *this;
        }
        eventComparatorConfig &set_mom(bool e, bool x, bool y, bool z)
        {
            compare_momentum = e || x || y || z;
            compare_momentum_E = e;
            compare_momentum_x = x;
            compare_momentum_y = y;
            compare_momentum_z = z;
            return *this;
        }
        eventComparatorConfig &set_E(bool v)
        {
            compare_momentum_E = v;
            compare_momentum = compare_momentum_x || compare_momentum_y || compare_momentum_z || compare_momentum_E;
            return *this;
        }
        eventComparatorConfig &set_t(bool v)
        {
            compare_momentum_E = v;
            compare_momentum = compare_momentum_x || compare_momentum_y || compare_momentum_z || compare_momentum_E;
            return *this;
        }
        eventComparatorConfig &set_x(bool v)
        {
            compare_momentum_x = v;
            compare_momentum = compare_momentum_x || compare_momentum_y || compare_momentum_z || compare_momentum_E;
            return *this;
        }
        eventComparatorConfig &set_px(bool v)
        {
            compare_momentum_x = v;
            compare_momentum = compare_momentum_x || compare_momentum_y || compare_momentum_z || compare_momentum_E;
            return *this;
        }
        eventComparatorConfig &set_y(bool v)
        {
            compare_momentum_y = v;
            compare_momentum = compare_momentum_x || compare_momentum_y || compare_momentum_z || compare_momentum_E;
            return *this;
        }
        eventComparatorConfig &set_py(bool v)
        {
            compare_momentum_y = v;
            compare_momentum = compare_momentum_x || compare_momentum_y || compare_momentum_z || compare_momentum_E;
            return *this;
        }
        eventComparatorConfig &set_z(bool v)
        {
            compare_momentum_z = v;
            compare_momentum = compare_momentum_x || compare_momentum_y || compare_momentum_z || compare_momentum_E;
            return *this;
        }
        eventComparatorConfig &set_pz(bool v)
        {
            compare_momentum_z = v;
            compare_momentum = compare_momentum_x || compare_momentum_y || compare_momentum_z || compare_momentum_E;
            return *this;
        }
        eventComparatorConfig &set_mass(bool v)
        {
            compare_mass = v;
            return *this;
        }
        eventComparatorConfig &set_m(bool v)
        {
            compare_mass = v;
            return *this;
        }
        eventComparatorConfig &set_mUP(bool v)
        {
            compare_mass = v;
            return *this;
        }
        eventComparatorConfig &set_vtim(bool v)
        {
            compare_vtim = v;
            return *this;
        }
        eventComparatorConfig &set_vTim(bool v)
        {
            compare_vtim = v;
            return *this;
        }
        eventComparatorConfig &set_vTimUP(bool v)
        {
            compare_vtim = v;
            return *this;
        }
        eventComparatorConfig &set_spin(bool v)
        {
            compare_spin = v;
            return *this;
        }
        eventComparatorConfig &set_spinUP(bool v)
        {
            compare_spin = v;
            return *this;
        }
        eventComparatorConfig &set_pdg(bool v)
        {
            compare_pdg = v;
            return *this;
        }
        eventComparatorConfig &set_id(bool v)
        {
            compare_pdg = v;
            return *this;
        }
        eventComparatorConfig &set_idUP(bool v)
        {
            compare_pdg = v;
            return *this;
        }
        eventComparatorConfig &set_status(bool v)
        {
            compare_status = v;
            return *this;
        }
        eventComparatorConfig &set_iSt(bool v)
        {
            compare_status = v;
            return *this;
        }
        eventComparatorConfig &set_iStUP(bool v)
        {
            compare_status = v;
            return *this;
        }
        eventComparatorConfig &set_mother(bool v)
        {
            compare_mother = v;
            return *this;
        }
        eventComparatorConfig &set_moth(bool v)
        {
            compare_mother = v;
            return *this;
        }
        eventComparatorConfig &set_mothUP(bool v)
        {
            compare_mother = v;
            return *this;
        }
        eventComparatorConfig &set_icol(bool v)
        {
            compare_icol = v;
            return *this;
        }
        eventComparatorConfig &set_iCol(bool v)
        {
            compare_icol = v;
            return *this;
        }
        eventComparatorConfig &set_iColUP(bool v)
        {
            compare_icol = v;
            return *this;
        }

        eventComparatorConfig &set_status_filter(std::vector<int> v)
        {
            status_filter = std::set<int>(v.begin(), v.end());
            return *this;
        }
        eventComparatorConfig &set_status_filter(std::set<int> s)
        {
            status_filter = std::move(s);
            return *this;
        }
        template <typename... Args>
        eventComparatorConfig &set_status_filter(Args... args)
        {
            status_filter = std::set<int>{args...};
            return *this;
        }
        eventComparatorConfig &set_tolerance(double tol)
        {
            mass_tol = tol;
            momentum_tol = tol;
            vtim_tol = tol;
            spin_tol = tol;
            weight_tol = tol;
            scale_tol = tol;
            alphaEW_tol = tol;
            alphaS_tol = tol;
            return *this;
        }
    };

    eventComparatorConfig compare_legs_only();
    eventComparatorConfig compare_final_state_only();
    eventComparatorConfig compare_physics_fields();

    using event_bool_fn = std::function<bool(event &)>;        // boolean function type
    using cevent_bool_fn = std::function<bool(const event &)>; // boolean function type

    // Struct to test whether an event belongs to a set of events
    struct eventBelongs
    {
    public:
        std::vector<std::shared_ptr<event>> events = {};
        event_equal_fn comparator = external_legs_comparator;
        cevent_equal_fn const_comparator = external_legs_const_comparator;
        // Default constructors
        eventBelongs() = default;
        eventBelongs(const eventBelongs &) = default;
        eventBelongs(eventBelongs &&) noexcept = default;
        eventBelongs &operator=(const eventBelongs &) = default;
        eventBelongs &operator=(eventBelongs &&) noexcept = default;
        // Constructor with one event
        explicit eventBelongs(const event &e);
        explicit eventBelongs(std::shared_ptr<event> e);
        // Constructor with multiple events
        explicit eventBelongs(std::vector<event> evts);
        explicit eventBelongs(std::vector<std::shared_ptr<event>> evts);
        // Constructors with comparator
        explicit eventBelongs(const event &e, event_equal_fn comp);
        explicit eventBelongs(std::shared_ptr<event> e, event_equal_fn comp);
        explicit eventBelongs(std::vector<event> evts, event_equal_fn comp);
        explicit eventBelongs(std::vector<std::shared_ptr<event>> evts, event_equal_fn comp);
        explicit eventBelongs(const event &e, cevent_equal_fn comp);
        explicit eventBelongs(std::shared_ptr<event> e, cevent_equal_fn comp);
        explicit eventBelongs(std::vector<event> evts, cevent_equal_fn comp);
        explicit eventBelongs(std::vector<std::shared_ptr<event>> evts, cevent_equal_fn comp);
        explicit eventBelongs(const event &e, event_equal_fn comp, cevent_equal_fn ccomp);
        explicit eventBelongs(std::shared_ptr<event> e, event_equal_fn comp, cevent_equal_fn ccomp);
        explicit eventBelongs(std::vector<event> evts, event_equal_fn comp, cevent_equal_fn ccomp);
        explicit eventBelongs(std::vector<std::shared_ptr<event>> evts, event_equal_fn comp, cevent_equal_fn ccomp);
        // Add an event to the set
        eventBelongs &add_event(const event &e);
        eventBelongs &add_event(std::shared_ptr<event> e);
        eventBelongs &add_event(const std::vector<event> &evts);
        eventBelongs &add_event(std::vector<std::shared_ptr<event>> evts);
        // Self-returning setting functions
        eventBelongs &set_events(const event &e);
        eventBelongs &set_events(std::shared_ptr<event> e);
        eventBelongs &set_events(const std::vector<event> &evts);
        eventBelongs &set_events(std::vector<std::shared_ptr<event>> evts);
        eventBelongs &set_comparator(event_equal_fn comp);
        eventBelongs &set_comparator(cevent_equal_fn comp);
        eventBelongs &set_comparator(const eventComparatorConfig &cfg);
        // Check if an event belongs to the set
        bool belongs_mutable(event &e);
        bool belongs_const(const event &e) const;
        bool belongs(std::shared_ptr<event> e);
        bool belongs(event &e);
        bool belongs(const event &e) const;
        // Overload parenthesis operator for easy usage
        bool operator()(event &e)
        {
            return belongs(e);
        }
        bool operator()(const event &e) const
        {
            return belongs(e);
        }
        event_bool_fn get_event_bool();
        cevent_bool_fn get_const_event_bool() const;
    };

    eventBelongs all_events_belong();

    using event_hash_fn = std::function<size_t(event &)>;        // hash function type
    using cevent_hash_fn = std::function<size_t(const event &)>; // hash function type

    // Struct to sort events by their belonging to sets of events
    struct eventSorter
    {
        std::vector<std::shared_ptr<eventBelongs>> event_sets = {};
        std::vector<event_bool_fn> comparators = {};
        std::vector<cevent_bool_fn> const_comparators = {};
        // Default constructors
        eventSorter() = default;
        eventSorter(const eventSorter &) = default;
        eventSorter(eventSorter &&) noexcept = default;
        eventSorter &operator=(const eventSorter &) = default;
        eventSorter &operator=(eventSorter &&) noexcept = default;
        // Constructor with one event set
        explicit eventSorter(const eventBelongs &e_set);
        explicit eventSorter(event_bool_fn comp);
        explicit eventSorter(cevent_bool_fn comp);
        explicit eventSorter(event_bool_fn comp, cevent_bool_fn ccomp);
        // Constructor with multiple event sets
        explicit eventSorter(std::vector<eventBelongs> e_sets);
        explicit eventSorter(std::vector<event_bool_fn> comps);
        explicit eventSorter(std::vector<event_bool_fn> comps, std::vector<cevent_bool_fn> ccomps);
        void extract_comparators();
        // Add an event set to the sorter
        eventSorter &add_event_set(const eventBelongs &e_set);
        eventSorter &add_event_set(const std::vector<eventBelongs> &e_sets);
        eventSorter &add_bool(event_bool_fn comp);
        eventSorter &add_const_bool(cevent_bool_fn comp);
        eventSorter &add_bool(event_bool_fn comp, cevent_bool_fn ccomp);
        eventSorter &add_bool(std::vector<event_bool_fn> comps);
        eventSorter &add_const_bool(std::vector<cevent_bool_fn> ccomps);
        eventSorter &add_bool(std::vector<event_bool_fn> comps, std::vector<cevent_bool_fn> ccomps);
        // Self-returning setting functions
        eventSorter &set_event_sets(const eventBelongs &e_set);
        eventSorter &set_event_sets(const std::vector<eventBelongs> &e_sets);
        eventSorter &set_bools(const event_bool_fn comp);
        eventSorter &set_const_bools(const cevent_bool_fn comp);
        eventSorter &set_bools(const event_bool_fn comp, const cevent_bool_fn ccomp);
        eventSorter &set_bools(const std::vector<event_bool_fn> comps);
        eventSorter &set_const_bools(const std::vector<cevent_bool_fn> comps);
        eventSorter &set_bools(const std::vector<event_bool_fn> comps, const std::vector<cevent_bool_fn> ccomps);
        // size() function just returns the number of event sets
        size_t size() const;
        // Function to find the position of an event in the sorter, returns npos if not found
        size_t position(event &e);
        size_t position(const event &e) const;
        size_t position(std::shared_ptr<event> e);
        std::vector<size_t> position(std::vector<event> &evts);
        std::vector<size_t> position(const std::vector<event> &evts) const;
        std::vector<size_t> position(std::vector<std::shared_ptr<event>> evts);
        std::vector<size_t> sort(std::vector<event> &evts);
        std::vector<size_t> sort(const std::vector<event> &evts) const;
        std::vector<size_t> sort(std::vector<std::shared_ptr<event>> evts);
        event_hash_fn get_hash();
        cevent_hash_fn get_const_hash() const;
    };

    eventSorter make_sample_sorter(std::vector<event> sample, event_equal_fn comp = external_legs_comparator);
    eventSorter make_sample_sorter(std::vector<std::shared_ptr<event>> sample, event_equal_fn comp = external_legs_comparator);

    struct process
    {
        // Default constructors
        process() = default;
        process(const process &) = default;
        process(process &&) noexcept = default;
        process &operator=(const process &) = default;
        process &operator=(process &&) noexcept = default;
        explicit process(std::vector<std::shared_ptr<event>> evts, bool filter_partons = false);
        explicit process(std::vector<event> evts, bool filter_partons = false);

        process &add_event(const event &e);
        process &add_event(std::shared_ptr<event> e);
        process &add_event_raw(const event &e);
        process &add_event_raw(std::shared_ptr<event> e);
        process &add_event_filtered(const event &e);
        process &add_event_filtered(std::shared_ptr<event> e);
        process &add_event(const std::vector<event> &evts);
        process &add_event(std::vector<std::shared_ptr<event>> evts);

        // LHEF data, vectorised
        // Note that vecors of vectors are not contiguous in memory
        std::vector<size_t> n_ = {};         // number of partons per event
        std::vector<size_t> n_summed = {};   // number of (summed) partons per event (ie n_summed[i] = sum(n[0:i]))
        std::vector<long int> proc_id_ = {}; // process IDs
        std::vector<double> weight_ = {};    // event weights
        std::vector<double> scale_ = {};     // event scales
        std::vector<double> muF_ = {};       // factorization scales
        std::vector<double> muR_ = {};       // renormalization scales
        std::vector<double> muPS_ = {};      // parton shower scales
        std::vector<double> alphaEW_ = {};   // electromagnetic coupling constants
        std::vector<double> alphaS_ = {};    // strong coupling constants
        vecArr4<double> momenta_ = {};
        std::vector<double> mass_ = {}, vtim_ = {}, spin_ = {};
        std::vector<long int> pdg_ = {};
        std::vector<short int> status_ = {};
        vecArr2<short int> mother_ = {}, icol_ = {};
        std::vector<std::vector<double>> wgts_ = {}; // additional weights, if any; note that wgt ids are not stored at the event level, so custom
        std::unordered_map<std::string, std::vector<std::any>> extra;

        bool filter = false; // whether to extract data using raw data or event_view

        // Accessors
        std::vector<size_t> &n() { return n_; }
        const std::vector<size_t> &n() const { return n_; }
        std::vector<size_t> &nUP() { return n_; }
        const std::vector<size_t> &nUP() const { return n_; }
        std::vector<long int> &idPrUP() { return proc_id_; }
        const std::vector<long int> &idPrUP() const { return proc_id_; }
        std::vector<long int> &proc_id() { return proc_id_; }
        const std::vector<long int> &proc_id() const { return proc_id_; }
        std::vector<long int> &idPr() { return proc_id_; }
        const std::vector<long int> &idPr() const { return proc_id_; }
        std::vector<double> &weight() { return weight_; }
        const std::vector<double> &weight() const { return weight_; }
        std::vector<double> &xWgtUP() { return weight_; }
        const std::vector<double> &xWgtUP() const { return weight_; }
        std::vector<double> &scale() { return scale_; }
        const std::vector<double> &scale() const { return scale_; }
        std::vector<double> &scalUP() { return scale_; }
        const std::vector<double> &scalUP() const { return scale_; }
        std::vector<double> &muF() { return muF_; }
        const std::vector<double> &muF() const { return muF_; }
        std::vector<double> &muR() { return muR_; }
        const std::vector<double> &muR() const { return muR_; }
        std::vector<double> &muPS() { return muPS_; }
        const std::vector<double> &muPS() const { return muPS_; }
        std::vector<double> &aQEDUP() { return alphaEW_; }
        const std::vector<double> &aQEDUP() const { return alphaEW_; }
        std::vector<double> &aQED() { return alphaEW_; }
        const std::vector<double> &aQED() const { return alphaEW_; }
        std::vector<double> &aEW() { return alphaEW_; }
        const std::vector<double> &aEW() const { return alphaEW_; }
        std::vector<double> &alphaEW() { return alphaEW_; }
        const std::vector<double> &alphaEW() const { return alphaEW_; }
        std::vector<double> &alphaS() { return alphaS_; }
        const std::vector<double> &alphaS() const { return alphaS_; }
        std::vector<double> &aQCDUP() { return alphaS_; }
        const std::vector<double> &aQCDUP() const { return alphaS_; }
        std::vector<double> &aQCD() { return alphaS_; }
        const std::vector<double> &aQCD() const { return alphaS_; }
        std::vector<double> &aS() { return alphaS_; }
        const std::vector<double> &aS() const { return alphaS_; }

        vecArr4<double> &pUP() { return momenta_; }
        const vecArr4<double> &pUP() const { return momenta_; }
        vecArr4<double> &mom() { return momenta_; }
        const vecArr4<double> &mom() const { return momenta_; }
        vecArr4<double> &p() { return momenta_; }
        const vecArr4<double> &p() const { return momenta_; }
        vecArr4<double> &momentum() { return momenta_; }
        const vecArr4<double> &momentum() const { return momenta_; }
        vecArr4<double> &momenta() { return momenta_; }
        const vecArr4<double> &momenta() const { return momenta_; }
        std::vector<double> &mUP() { return mass_; }
        const std::vector<double> &mUP() const { return mass_; }
        std::vector<double> &m() { return mass_; }
        const std::vector<double> &m() const { return mass_; }
        std::vector<double> &mass() { return mass_; }
        const std::vector<double> &mass() const { return mass_; }
        std::vector<double> &vtim() { return vtim_; }
        const std::vector<double> &vtim() const { return vtim_; }
        std::vector<double> &vTimUP() { return vtim_; }
        const std::vector<double> &vTimUP() const { return vtim_; }
        std::vector<double> &vTim() { return vtim_; }
        const std::vector<double> &vTim() const { return vtim_; }
        std::vector<double> &spin() { return spin_; }
        const std::vector<double> &spin() const { return spin_; }
        std::vector<double> &spinUP() { return spin_; }
        const std::vector<double> &spinUP() const { return spin_; }
        std::vector<long int> &idUP() { return pdg_; }
        const std::vector<long int> &idUP() const { return pdg_; }
        std::vector<long int> &pdg() { return pdg_; }
        const std::vector<long int> &pdg() const { return pdg_; }
        std::vector<long int> &id() { return pdg_; }
        const std::vector<long int> &id() const { return pdg_; }
        std::vector<short int> &status() { return status_; }
        const std::vector<short int> &status() const { return status_; }
        std::vector<short int> &iSt() { return status_; }
        const std::vector<short int> &iSt() const { return status_; }
        std::vector<short int> &iStUP() { return status_; }
        const std::vector<short int> &iStUP() const { return status_; }
        vecArr2<short int> &mother() { return mother_; }
        const vecArr2<short int> &mother() const { return mother_; }
        vecArr2<short int> &moth() { return mother_; }
        const vecArr2<short int> &moth() const { return mother_; }
        vecArr2<short int> &mothUP() { return mother_; }
        const vecArr2<short int> &mothUP() const { return mother_; }
        vecArr2<short int> &iColUP() { return icol_; }
        const vecArr2<short int> &iColUP() const { return icol_; }
        vecArr2<short int> &iCol() { return icol_; }
        const vecArr2<short int> &iCol() const { return icol_; }
        vecArr2<short int> &icol() { return icol_; }
        const vecArr2<short int> &icol() const { return icol_; }

        std::vector<std::vector<double>> &wgtUP() { return wgts_; }
        const std::vector<std::vector<double>> &wgtUP() const { return wgts_; }
        std::vector<std::vector<double>> &wgts() { return wgts_; }
        const std::vector<std::vector<double>> &wgts() const { return wgts_; }

        std::vector<double> &get_muF()
        {
            // if muF empty, set it equal to scale before returning
            if (muF_.empty())
                muF_ = scale_;
            return muF_;
        }

        std::vector<double> &get_muR()
        {
            // if muR empty, set it equal to scale before returning
            if (muR_.empty())
                muR_ = scale_;
            return muR_;
        }

        std::vector<double> &get_muPS()
        {
            // if muPS empty, set it equal to scale before returning
            if (muPS_.empty())
                muPS_ = scale_;
            return muPS_;
        }

        // Specific momenta components --- not references!
        // When accessing specific momentum components like this,
        // the data is copied and then needs to be overwritten
        // in the original process object using the set_* functions
        std::vector<double> E();
        std::vector<double> t();
        std::vector<double> x();
        std::vector<double> px();
        std::vector<double> y();
        std::vector<double> py();
        std::vector<double> z();
        std::vector<double> pz();
        process &set_E(const std::vector<double> &E);
        process &set_t(const std::vector<double> &pt);
        process &set_x(const std::vector<double> &x);
        process &set_px(const std::vector<double> &px);
        process &set_y(const std::vector<double> &y);
        process &set_py(const std::vector<double> &py);
        process &set_z(const std::vector<double> &z);
        process &set_pz(const std::vector<double> &pz);

        std::vector<double> gS();
        process &set_gS(const std::vector<double> &gS);

        // vector of events (can be overwritten!)
        // For LHE objects, the events are shared between
        // the process objects and the owning LHE object
        std::vector<std::shared_ptr<event>> events = {};

        // Self-returning setting functions
        process &set_n(const std::vector<size_t> &n);
        process &set_n_summed(const std::vector<size_t> &n_summed);
        process &set_proc_id(const std::vector<long int> &proc_id);
        process &set_weight(const std::vector<double> &weight);
        process &set_scale(const std::vector<double> &scale);
        process &set_muF(const std::vector<double> &muF);
        process &set_muR(const std::vector<double> &muR);
        process &set_muPS(const std::vector<double> &muPS);
        process &set_alphaEW(const std::vector<double> &alphaEW);
        process &set_alphaS(const std::vector<double> &alphaS);
        process &set_momenta(const vecArr4<double> &momenta);
        process &set_mass(const std::vector<double> &mass);
        process &set_vtim(const std::vector<double> &vtim);
        process &set_spin(const std::vector<double> &spin);
        process &set_pdg(const std::vector<long int> &pdg);
        process &set_status(const std::vector<short int> &status);
        process &set_mother(const vecArr2<short int> &mother);
        process &set_icol(const vecArr2<short int> &icol);
        process &set_wgts(const std::vector<std::vector<double>> &wgts);
        process &append_wgts(const std::vector<double> &wgts);
        process &add_extra(const std::string &name, const std::any &value);
        process &add_extra(const std::unordered_map<std::string, std::any> &values);
        process &add_extra(const std::string &name, const std::vector<std::any> &values);
        process &add_extra(const std::unordered_map<std::string, std::vector<std::any>> &values);
        process &set_extra(const std::unordered_map<std::string, std::vector<std::any>> &values);

        // Flag whether to pull event data according to internal
        // storage or to event_view indexing
        process &set_filter(bool v);

        // Throws if any vector has a mismatched size with each other or nUP
        void validate() const;

        // Functions for total transposition to the events vector
        void make_event(size_t idx);
        void transpose();

        // Partial transposition functions, moving a single parameter into each event in the events vector
        process &transpose_n();
        process &transpose_nUP();
        process &transpose_proc_id();
        process &transpose_idPrUP();
        process &transpose_idPr();
        process &transpose_weight();
        process &transpose_xWgtUP();
        process &transpose_xWgt();
        process &transpose_scale();
        process &transpose_scalUP();
        process &transpose_muF();
        process &transpose_muR();
        process &transpose_muPS();
        process &transpose_alphaEW();
        process &transpose_aQEDUP();
        process &transpose_aQED();
        process &transpose_alphaS();
        process &transpose_aQCDUP();
        process &transpose_aQCD();
        process &transpose_momenta();
        process &transpose_pUP();
        process &transpose_mom();
        process &transpose_p();
        process &transpose_mass();
        process &transpose_mUP();
        process &transpose_m();
        process &transpose_vtim();
        process &transpose_vTimUP();
        process &transpose_spin();
        process &transpose_spinUP();
        process &transpose_pdg();
        process &transpose_idUP();
        process &transpose_id();
        process &transpose_status();
        process &transpose_iStUP();
        process &transpose_iSt();
        process &transpose_mother();
        process &transpose_mothUP();
        process &transpose_icol();
        process &transpose_iColUP();
        process &transpose_wgts();
        process &transpose_extra();

        // Specific momentum component transpositions
        process &transpose_E();
        process &transpose_t();
        process &transpose_x();
        process &transpose_px();
        process &transpose_y();
        process &transpose_py();
        process &transpose_z();
        process &transpose_pz();
    };

    // Init class
    // Primarily used to make it possible to split init node information
    // from the rest of the LHE information without needing to
    // complicate the logic in readers and writers
    struct initNode
    {
        // Default constructors
        initNode() = default;
        initNode(const initNode &) = default;
        initNode(initNode &&) noexcept = default;
        initNode &operator=(const initNode &) = default;
        initNode &operator=(initNode &&) noexcept = default;
        // Custom constructors
        initNode(short unsigned int nproc);
        initNode(size_t nproc);

        arr2<long int> idBm_ = {0, 0};  // beam IDs
        arr2<double> eBm_ = {0.0, 0.0}; // beam energies
        arr2<short int> pdfG_ = {0, 0}; // PDF group IDs
        arr2<long int> pdfS_ = {0, 0};  // PDF set IDs
        short int idWgt_ = 0;           // weight ID
        short unsigned int nProc_ = 0;  // number of processes

        std::vector<double> xSec_ = {};    // cross sections
        std::vector<double> xSecErr_ = {}; // cross section errors
        std::vector<double> xMax_ = {};    // maximum weights
        std::vector<long int> lProc_ = {}; // process IDs

        // Access functions for various alternative names of the variables above, all passed as references
        arr2<long int> &idBmUP() { return idBm_; }
        const arr2<long int> &idBmUP() const { return idBm_; }
        arr2<long int> &idBm() { return idBm_; }
        const arr2<long int> &idBm() const { return idBm_; }
        arr2<double> &eBmUP() { return eBm_; }
        const arr2<double> &eBmUP() const { return eBm_; }
        arr2<double> &eBm() { return eBm_; }
        const arr2<double> &eBm() const { return eBm_; }
        arr2<short int> &pdfGUP() { return pdfG_; }
        const arr2<short int> &pdfGUP() const { return pdfG_; }
        arr2<short int> &pdfG() { return pdfG_; }
        const arr2<short int> &pdfG() const { return pdfG_; }
        arr2<long int> &pdfSUP() { return pdfS_; }
        const arr2<long int> &pdfSUP() const { return pdfS_; }
        arr2<long int> &pdfS() { return pdfS_; }
        const arr2<long int> &pdfS() const { return pdfS_; }
        short int &idWgtUP() { return idWgt_; }
        const short int &idWgtUP() const { return idWgt_; }
        short int &idWgt() { return idWgt_; }
        const short int &idWgt() const { return idWgt_; }
        short unsigned int &nProcUP() { return nProc_; }
        const short unsigned int &nProcUP() const { return nProc_; }
        short unsigned int &nProc() { return nProc_; }
        const short unsigned int &nProc() const { return nProc_; }
        std::vector<double> &xSecUP() { return xSec_; }
        const std::vector<double> &xSecUP() const { return xSec_; }
        std::vector<double> &xSec() { return xSec_; }
        const std::vector<double> &xSec() const { return xSec_; }
        std::vector<double> &xSecErrUP() { return xSecErr_; }
        const std::vector<double> &xSecErrUP() const { return xSecErr_; }
        std::vector<double> &xSecErr() { return xSecErr_; }
        const std::vector<double> &xSecErr() const { return xSecErr_; }
        std::vector<double> &xMaxUP() { return xMax_; }
        const std::vector<double> &xMaxUP() const { return xMax_; }
        std::vector<double> &xMax() { return xMax_; }
        const std::vector<double> &xMax() const { return xMax_; }
        std::vector<long int> &lProcUP() { return lProc_; }
        const std::vector<long int> &lProcUP() const { return lProc_; }
        std::vector<long int> &lProc() { return lProc_; }
        const std::vector<long int> &lProc() const { return lProc_; }

        initNode &set_idBm(const arr2<long int> &ids);
        initNode &set_idBm(long int id1, long int id2);
        initNode &set_eBm(const arr2<double> &energies);
        initNode &set_eBm(double e1, double e2);
        initNode &set_pdfG(const arr2<short int> &ids);
        initNode &set_pdfG(short int id1, short int id2);
        initNode &set_pdfS(const arr2<long int> &ids);
        initNode &set_pdfS(long int id1, long int id2);
        initNode &set_idWgt(short int id);
        initNode &set_nProc(short unsigned int n);
        initNode &set_xSec(const std::vector<double> &xsec);
        initNode &set_xSecErr(const std::vector<double> &xsec_err);
        initNode &set_xMax(const std::vector<double> &xmax);
        initNode &set_lProc(const std::vector<long int> &lproc);
        initNode &add_xSec(double xsec);
        initNode &add_xSecErr(double xsec_err);
        initNode &add_xMax(double xmax);
        initNode &add_lProc(long int lproc);

        void validate_init() const;

        void print_head(std::ostream &os = std::cout) const;
        void print_body(std::ostream &os = std::cout) const;
        void print_extra(std::ostream &os = std::cout) const;
        void print_init(std::ostream &os = std::cout) const;

        // Generic additional tags/information
        std::unordered_map<std::string, std::any> extra;

        // Set generic data (overwrites if exists)
        // Note: std::any is used to allow any type of value to be stored
        // This is a simple key-value store for data, not part of the LHEF standard
        template <typename T>
        void set(const std::string &name, T value)
        {
            extra[name] = std::any(std::move(value));
        }

        // Get extra data (throws if not found or wrong type)
        template <typename T>
        T &get(const std::string &name)
        {
            auto it = extra.find(name);
            if (it == extra.end())
            {
                throw std::out_of_range("No parameter named '" + name + "'");
            }
            return std::any_cast<T &>(it->second);
        }

        template <typename T>
        const T &get(const std::string &name) const
        {
            auto it = extra.find(name);
            if (it == extra.end())
            {
                throw std::out_of_range("No parameter named '" + name + "'");
            }
            return std::any_cast<const T &>(it->second);
        }

        bool has(const std::string &name) const
        {
            return extra.find(name) != extra.end();
        }
    };

    // Data-driven LHE struct
    // Contains both the object oriented event representation
    // and the data-oriented SoA process representation,
    // plus the additional information stored in the init node
    // Does not have a defined header object aside from a generic std::any which
    // needs to be unpacked manually if used, although weight ids as in the initrwgt node
    // can be stored in a vector of strings (without any weight group splittings)
    // and can be passed on to the event nodes for writing weights with ids in the <rwgt> format
    struct lhe : public initNode
    {
        // Default constructors
        lhe() = default;
        lhe(const lhe &) = default;
        lhe(lhe &&) noexcept = default;
        lhe &operator=(const lhe &) = default;
        lhe &operator=(lhe &&) noexcept = default;
        // Custom constructors
        explicit lhe(const initNode &i) : initNode(i) {};
        explicit lhe(std::vector<std::shared_ptr<event>> evts);
        explicit lhe(std::vector<event> evts);
        explicit lhe(const initNode &i, std::vector<std::shared_ptr<event>> evts);
        explicit lhe(const initNode &i, std::vector<event> evts);

        std::any header;

        bool filter_processes = true;

        eventSorter sorter;
        event_hash_fn event_hash = nullptr; // hash function for events (ie sorter)

        std::vector<size_t> process_order = {}; // mapping from events to processes

        std::vector<std::shared_ptr<event>> events = {};                // vector of events
        std::vector<std::vector<std::shared_ptr<event>>> sorted_events; // vector of vectors of events, sorted according to the processes scheme
        std::vector<std::shared_ptr<process>> processes = {};           // vector of processes

        std::shared_ptr<std::vector<std::string>> weight_ids = std::make_shared<std::vector<std::string>>(); // weight ids for the <rwgt> block, if any

        std::vector<std::string> weight_context = {}; // context strings for each weight, if any

        // Self-returning setting functions
        lhe &set_events(std::vector<std::shared_ptr<event>> evts);
        lhe &set_processes(std::vector<std::shared_ptr<process>> procs);
        lhe &set_header(std::any hdr);
        lhe &add_event(std::shared_ptr<event> evt);
        lhe &add_event(const event &evt);
        lhe &set_sorter();
        lhe &set_sorter(event_equal_fn comp);
        lhe &set_sorter(cevent_equal_fn comp);
        lhe &set_sorter(const eventSorter &s);
        void extract_hash();
        lhe &set_hash(event_hash_fn hash);
        lhe &set_filter(bool v);
        lhe &set_weight_ids(const std::vector<std::string> &ids);
        lhe &set_weight_ids(std::vector<std::string> &&ids);
        lhe &set_weight_ids(std::shared_ptr<std::vector<std::string>> ids);
        lhe &add_weight_id(const std::string &id);
        lhe &add_weight_id(std::string &&id);

        void sort_events_mutable();
        void sort_events_const();

        void sort_events();
        void unsort_events();
        void events_to_processes();
        void processes_to_events();
        void transpose();
        void transpose(std::string dir);

        void extract_weight_ids();
        void sync_weight_ids();
        void append_weight_ids(bool include = false);

        void print_header(std::ostream &os = std::cout) const;
        void print(std::ostream &os = std::cout, bool include_ids = false);
    };

    // Classes for XML handling
    // xmlDoc: Owns the entire XML text buffer. Nodes keep this alive via shared_ptr.
    class xmlDoc
    {
    public:
        xmlDoc() = default;
        explicit xmlDoc(std::string xml);
        const std::string &str() const noexcept;
        std::string_view view() const noexcept;
        std::shared_ptr<const std::string> shared() const noexcept;

    private:
        std::shared_ptr<std::string> buf_{std::make_shared<std::string>()};
    };

    // Attribute with copy-on-write mutability.
    struct Attr
    {
        // Original views into the mother buffer:
        std::string_view name_view{};
        std::string_view value_view{};

        // Only allocated if modified:
        std::optional<std::string> name_new{};
        std::optional<std::string> value_new{};

        std::string_view name() const noexcept;
        std::string_view value() const noexcept;
        bool modified() const noexcept;
    };

    // xmlNode: Main XML node representation
    // Handles parsing, writing, access etc
    class xmlNode
    {
    public:
        xmlNode();
        ~xmlNode();

        // Parse from an owning string (keeps only one owned copy).
        static std::shared_ptr<xmlNode> parse(std::string xml);

        // Parse from an already-shared buffer.
        static std::shared_ptr<xmlNode> parse(const std::shared_ptr<const std::string> &buf);

        std::string_view name() const noexcept;
        std::string_view full() const noexcept;    // full slice [start_, end_)
        std::string_view content() const noexcept; // [content_start_, content_end_)
        const std::vector<Attr> &attrs() const noexcept;
        std::vector<std::shared_ptr<xmlNode>> &children();

        bool modified(bool deep = false) const noexcept;
        bool is_leaf() const noexcept;

        // Offsets into the mother buffer:
        size_t start() const noexcept { return start_; }
        size_t head_end() const noexcept { return head_end_; }
        size_t content_start() const noexcept { return content_start_; }
        size_t content_end() const noexcept { return content_end_; }
        size_t end() const noexcept { return end_; }

        // ---- Mutations (copy-on-write) ----
        void set_name(std::string new_name);
        void set_content(std::string new_content);                           // replace text content
        void set_content_writer(std::function<void(std::ostream &)> writer); // stream replacement
        bool set_attr(std::string_view key, std::string new_value);          // returns false if not found
        void add_attr(std::string name, std::string value);                  // adds a brand-new attribute

        void add_child(std::shared_ptr<xmlNode> child, bool add_nl = false);

        void write(std::ostream &os) const; // zero-copy where possible
        void write(std::string &out) const; // appends into out

        size_t n_children() const noexcept { return children_.size(); }

        bool has_child(std::string_view name) const noexcept;
        std::shared_ptr<xmlNode> get_child(std::string_view name) const noexcept; // first matching child or nullptr

        std::vector<std::shared_ptr<xmlNode>> get_children(std::string_view name) const noexcept; // all matching children

        std::shared_ptr<xmlNode> deep_copy() const;

        // Remove/suppress a child so it won’t be written (but offsets still used to skip its original bytes)
        bool remove_child(size_t index) noexcept;
        bool remove_child(const xmlNode *child) noexcept;
        bool remove_child(std::string_view name) noexcept;

        // Insert new children relative to existing ones
        bool insert_child_before(size_t anchor_in_doc_ordinal, std::shared_ptr<xmlNode> child) noexcept;
        bool insert_child_after(size_t anchor_in_doc_ordinal, std::shared_ptr<xmlNode> child) noexcept;

        // Replace an existing child at same spot (suppresses old, inserts new before it)
        bool replace_child(size_t anchor_in_doc_ordinal, std::shared_ptr<xmlNode> child) noexcept;
        bool replace_child(std::string_view anchor_name, std::shared_ptr<xmlNode> child, bool add_nl = false) noexcept;

        // Insert at specific location in this node's content
        // (rel_offset is in bytes relative to content_start(); clamped to [0, content_len]).
        bool insert_child_at_content_offset(size_t rel_offset, std::shared_ptr<xmlNode> child) noexcept;

        // Convenience: explicit start/end of content
        bool insert_child_at_start(std::shared_ptr<xmlNode> child) noexcept;
        bool insert_child_at_end(std::shared_ptr<xmlNode> child) noexcept;

        // Hints to add newlines before starting or ending nodes
        bool append_nl_start = false;
        bool append_nl_end = false;

    private:
        // Internal constructor used by parser.
        explicit xmlNode(std::shared_ptr<const std::string> doc);

        // Recursive element parser. Expects 'pos' at '<' for a normal element.
        static std::shared_ptr<xmlNode> parse_element(const std::shared_ptr<const std::string> &doc, size_t &pos);

        // Top-level scanner to locate the first element (skips XML decl, comments, etc.).
        static size_t find_first_element_start(const std::string &s, size_t pos);

        // Helpers for parsing attributes and skipping markup we don't turn into nodes.
        static void parse_attributes(xmlNode &node, size_t &cur);
        static bool skip_comment(const std::string &s, size_t &pos); // <!-- ... -->
        static bool skip_pi(const std::string &s, size_t &pos);      // <? ... ?>
        static bool skip_doctype(const std::string &s, size_t &pos); // <!DOCTYPE ...>
        static bool skip_cdata(const std::string &s, size_t &pos);   // <![CDATA[ ... ]]>

        // Writer helpers
        void write_start_tag(std::ostream &os) const;
        void write_end_tag(std::ostream &os) const;
        bool modified_header() const noexcept;
        bool modified_footer() const noexcept;

        std::shared_ptr<const std::string> doc_{};

        // Offsets into *doc_:
        size_t start_ = npos;
        size_t head_end_ = npos;
        size_t content_start_ = npos;
        size_t content_end_ = npos;
        size_t end_ = npos;

        size_t prolog_start_ = 0; // usually 0
        size_t prolog_end_ = 0;   // byte offset of first '<' of the root element

        // Child nodes may be “suppressed” for serialization by a parent;
        // this flag is *only* consulted by the parent’s writer loop.
        bool suppressed_ = false;

        bool self_closing_ = false;

        // Internal: write with context (as a root or embedded as a child)
        void write_impl(std::ostream &os, bool as_child) const;

        const xmlNode *nth_in_doc_child(size_t ordinal) const noexcept;

        // Placement hints for synthetic (added) children; ownership stays in children_
        struct InsertHint
        {
            enum class Where
            {
                Before,
                After,
                AtAbs,
                AtStart,
                AtEnd
            };
            Where where;
            const xmlNode *anchor = nullptr; // for Before/After
            size_t abs = 0;                  // absolute byte offset in this->doc_ for AtAbs
            const xmlNode *node = nullptr;   // child to write (owned by children_)
        };
        std::vector<InsertHint> inserts_;

        std::string_view name_view_{};
        std::optional<std::string> name_new_{};
        std::optional<std::string> content_new_{};
        std::function<void(std::ostream &)> content_writer_{};
        std::vector<Attr> attrs_{};
        std::vector<std::shared_ptr<xmlNode>> children_{};
        bool modified_ = false;
    };

    // Helpers for lheReader
    namespace lhe_build_detail
    {

        template <class T>
        inline constexpr bool is_shared_event_v =
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::shared_ptr<event>>;

        template <class T>
        inline constexpr bool is_unique_event_v =
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::unique_ptr<event>>;

        template <class T>
        inline constexpr bool is_event_object_v =
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, event> ||
            std::is_base_of_v<event, std::remove_cv_t<std::remove_reference_t<T>>>;

        // Convert {event / unique_ptr / shared_ptr} -> shared_ptr<event>
        template <class U>
        std::shared_ptr<event> to_shared_event(U &&u)
        {
            if constexpr (is_shared_event_v<U>)
            {
                return std::forward<U>(u);
            }
            else if constexpr (is_unique_event_v<U>)
            {
                // move out of unique_ptr into shared_ptr
                auto *raw = std::forward<U>(u).release();
                return std::shared_ptr<event>(raw);
            }
            else if constexpr (is_event_object_v<U>)
            {
                using Decayed = std::remove_cv_t<std::remove_reference_t<U>>;
                if constexpr (std::is_same_v<Decayed, event>)
                {
                    return std::make_shared<event>(std::forward<U>(u));
                }
                else
                {
                    // Derived from event
                    return std::make_shared<Decayed>(std::forward<U>(u));
                }
            }
            else
            {
                static_assert(is_shared_event_v<U> || is_unique_event_v<U> || is_event_object_v<U>,
                              "Event translator must return event, unique_ptr<event>, or shared_ptr<event>.");
                return {}; // unreachable
            }
        }

    } // namespace lhe_build_detail

    // lheReader: Builds LHE (Les Houches Event) files from raw inputs
    // takes as input an initNode reader, an event reader, and an optional header reader
    template <class InitRaw, class EventRaw, class HeaderRaw = std::monostate>
    class lheReader
    {
    public:
        using InitTx = std::function<initNode(const InitRaw &)>;
        using EventTx = std::function<std::shared_ptr<event>(const EventRaw &)>;
        using HeaderTx = std::function<std::any(const HeaderRaw &)>;

        lheReader() = default;
        lheReader(const lheReader &) = default;
        lheReader(lheReader &&) = default;
        lheReader &operator=(const lheReader &) = default;
        lheReader &operator=(lheReader &&) = default;
        template <class InitTx, class EventTx>
        explicit lheReader(InitTx init_tx, EventTx event_tx)
        {
            set_init_translator(std::move(init_tx));
            set_event_translator(std::move(event_tx));
        }

        template <class InitTx, class EventTx, class HeaderTx>
        lheReader(InitTx init_tx, EventTx event_tx, HeaderTx header_tx)
        {
            set_init_translator(std::move(init_tx));
            set_event_translator(std::move(event_tx));
            set_header_translator(std::move(header_tx));
        }

        lheReader &set_init_translator(InitTx tx)
        {
            init_tx_ = std::move(tx);
            return *this;
        }

        template <class F>
        lheReader &set_event_translator(F &&f)
        {
            // Accepts any callable returning event-like; adapt to shared_ptr<event>.
            event_tx_ = [fn = std::forward<F>(f)](const EventRaw &r) -> std::shared_ptr<event>
            {
                auto out = fn(r);
                return lhe_build_detail::to_shared_event(std::move(out));
            };
            return *this;
        }

        lheReader &set_header_translator(HeaderTx tx)
        {
            header_tx_ = std::move(tx);
            return *this;
        }

        lheReader &set_filter(bool v)
        {
            filter_processes_ = v;
            return *this;
        }

        // Read from raw inputs
        template <class EventRange>
        lhe read(const InitRaw &init_raw,
                 const EventRange &events_raw,
                 std::optional<HeaderRaw> header_raw = std::nullopt) const
        {
            ensure_ready_();

            // 1) Read initNode and validate
            initNode init = init_tx_(init_raw);
            init.validate_init();

            // 2) Translate events -> vector<shared_ptr<event>>
            std::vector<std::shared_ptr<event>> evts;
            if constexpr (has_size_v<EventRange>)
            {
                evts.reserve(events_raw.size());
            }
            for (const auto &er : events_raw)
            {
                evts.emplace_back(event_tx_(er));
                if (!evts.back())
                {
                    throw std::runtime_error("event translator produced null shared_ptr<event>.");
                }
            }

            // 3) Construct lhe with init + events
            lhe out(init, std::move(evts));

            // 4) Optional header
            if (header_raw.has_value())
            {
                if (!header_tx_)
                {
                    throw std::runtime_error("Header provided but no header translator configured.");
                }
                out.set_header(header_tx_.value()(header_raw.value()));
            }

            out.set_filter(filter_processes_);
            return out;
        }

    private:
        void ensure_ready_() const
        {
            if (!init_tx_)
                throw std::runtime_error("init translator not set.");
            if (!event_tx_)
                throw std::runtime_error("event translator not set.");
        }

        // Translators
        InitTx init_tx_;
        EventTx event_tx_;
        std::optional<HeaderTx> header_tx_;

        bool filter_processes_ = false;
    };

    template <class InitRaw,
              class EventRange,
              class HeaderRaw = std::monostate,
              class InitTx,
              class EventTx,
              class HeaderTx = std::nullptr_t>
    lhe read_lhe(const InitRaw &init_raw,
                 const EventRange &events_raw,
                 InitTx init_tx,
                 EventTx event_tx,
                 std::optional<HeaderRaw> header_raw = std::nullopt,
                 HeaderTx header_tx = nullptr,
                 bool filter_processes = false)
    {
        using EventRawT = typename std::decay<decltype(*std::begin(events_raw))>::type;

        lheReader<InitRaw, EventRawT, HeaderRaw> b;
        b.set_init_translator(std::move(init_tx))
            .set_event_translator(std::move(event_tx))
            .set_filter(filter_processes);

        if constexpr (!std::is_same_v<HeaderTx, std::nullptr_t>)
        {
            if constexpr (std::is_pointer_v<std::decay_t<HeaderTx>>)
            {
                if (header_tx)
                    b.set_header_translator(header_tx);
            }
            else
            {
                b.set_header_translator(std::move(header_tx));
            }
        }
        return b.read(init_raw, events_raw, header_raw);
    }

    struct ignore_header_t
    {
        std::optional<std::monostate> operator()(const std::any &) const noexcept
        {
            return std::nullopt;
        }
    };

    // output container post writing to user-defined class
    template <class RawInit, class RawEvent, class RawHeaderOpt>
    struct lheRaw
    {
        using init_type = RawInit;
        using event_type = RawEvent;
        using header_opt = RawHeaderOpt;

        RawInit init;
        std::vector<RawEvent> events;
        RawHeaderOpt header;
    };

    // lheWriter: Takes user-supplied constructors and applies them to
    // the lhe object and returns an lheRaw container from it
    template <
        class InitRaw,
        class EventRaw,
        class HeaderRaw = std::monostate>
    class lheWriter
    {
    public:
        using InitTx = std::function<InitRaw(const initNode &)>;
        using EventTx = std::function<EventRaw(event &)>;
        using HeaderTx = std::function<HeaderRaw(const std::any &)>;
        using result_t = lheRaw<InitRaw, EventRaw, HeaderRaw>;

        lheWriter(InitTx init_tx, EventTx event_tx, HeaderTx header_tx = HeaderTx{})
            : init_fn_(std::move(init_tx)), event_fn_(std::move(event_tx)), header_fn_(std::move(header_tx)) {}

        lheWriter &set_init_translator(InitTx tx)
        {
            init_fn_ = std::move(tx);
            return *this;
        }

        lheWriter &set_event_translator(EventTx tx)
        {
            event_fn_ = std::move(tx);
            return *this;
        }

        lheWriter &set_header_translator(HeaderTx tx)
        {
            header_fn_ = std::move(tx);
            return *this;
        }

        result_t to_raw(const lhe &doc) const
        {
            result_t out;

            if (doc.header.has_value())
            {
                if (header_fn_)
                    out.header = header_fn_(doc.header);
                else
                    warning("lheWriter::to_raw(): header present but no header translator configured; ignoring.");
            }

            out.init = init_fn_(static_cast<const initNode &>(doc));

            out.events.reserve(doc.events.size());
            for (const auto &pevt : doc.events)
                if (pevt)
                    out.events.push_back(event_fn_(*pevt));

            return out;
        }

        // Convenience: build raw, then let any writer consume it (kept separate by design)
        template <class OS, class Writer>
        void write(const lhe &doc, OS &os, Writer &&writer) const
        {
            auto raw = to_raw(doc);
            std::forward<Writer>(writer)(os, raw);
        }

    private:
        InitTx init_fn_;
        EventTx event_fn_;
        HeaderTx header_fn_;
    };

    template <class InitRaw,
              class EventRange,
              class EventRawT = typename std::decay<decltype(*std::begin(std::declval<EventRange>()))>::type,
              class HeaderRaw = std::monostate,
              class InitTx,
              class EventTx,
              class HeaderTx = std::nullptr_t>
    lheRaw<InitRaw, EventRawT, HeaderRaw> write_lhe(lhe &doc,
                                                    InitTx init_tx,
                                                    EventTx event_tx,
                                                    HeaderTx header_tx = nullptr)
    {
        // using EventRawT = typename std::decay<decltype(*std::begin(event_tx(doc.events[0])))>::type;

        lheWriter<InitRaw, EventRawT, HeaderRaw> w(std::move(init_tx), std::move(event_tx));
        if constexpr (!std::is_same_v<HeaderTx, std::nullptr_t>)
        {
            w.set_header_translator(std::move(header_tx));
        }
        return w.to_raw(doc);
    }

    std::shared_ptr<event> string_to_event(std::string_view content);
    std::shared_ptr<event> xml_to_event(std::shared_ptr<xmlNode> node);
    initNode string_to_init(std::string_view content);
    initNode xml_to_init(std::shared_ptr<xmlNode> node);
    std::any xml_to_any(std::shared_ptr<xmlNode> node);
    template <typename T>
    std::any to_any(T &&value)
    {
        return std::make_any<std::decay_t<T>>(std::forward<T>(value));
    }

    std::shared_ptr<xmlNode> init_to_xml(const initNode &);
    std::shared_ptr<xmlNode> event_to_xml(event &);
    std::optional<std::shared_ptr<xmlNode>> header_to_xml(const std::any &);

    using XmlToInitFn = initNode (*)(std::shared_ptr<xmlNode>);
    using XmlToEventFn = std::shared_ptr<event> (*)(std::shared_ptr<xmlNode>);
    using XmlToHeaderFn = std::any (*)(std::shared_ptr<xmlNode>);
    using InitToXmlFn = std::shared_ptr<xmlNode> (*)(const initNode &);
    using EventToXmlFn = std::shared_ptr<xmlNode> (*)(event &);
    using HeaderToXmlFn = std::optional<std::shared_ptr<xmlNode>> (*)(const std::any &);

    using xmlReader = lheReader<std::shared_ptr<xmlNode>,
                                std::shared_ptr<xmlNode>,
                                std::shared_ptr<xmlNode>>;
    using xmlWriter = lheWriter<
        std::shared_ptr<xmlNode>,               // InitRaw
        std::shared_ptr<xmlNode>,               // EventRaw
        std::optional<std::shared_ptr<xmlNode>> // HeaderRaw (must be optional<...>)
        >;

    using xmlRaw = lheRaw<
        std::shared_ptr<xmlNode>,               // RawInit
        std::shared_ptr<xmlNode>,               // RawEvent
        std::optional<std::shared_ptr<xmlNode>> // RawHeaderOpt
        >;

    // Accessor to prebuilt XML instance
    const xmlReader &xml_reader();
    const xmlWriter &xml_writer();

    // Convenience wrapper: translate to the Raw structure
    xmlRaw to_xml_raw(const lhe &doc);
    extern template class lheWriter<
        std::shared_ptr<xmlNode>,
        std::shared_ptr<xmlNode>,
        std::optional<std::shared_ptr<xmlNode>>>;
    template class lheWriter<
        std::shared_ptr<xmlNode>,
        std::shared_ptr<xmlNode>,
        std::optional<std::shared_ptr<xmlNode>>>;

    lhe to_lhe(std::shared_ptr<xmlNode> node);
    lhe to_lhe(const std::string &xml);
    lhe load_lhef(std::istream &in);
    lhe load_lhef(const std::string &filename);
    void write_lhef(lhe &doc, std::ostream &out = std::cout, bool include_ids = false);
    void write_lhef(lhe &doc, const std::string &filename, bool include_ids = false);
    std::shared_ptr<xmlNode> to_xml(xmlRaw &raw);
    std::shared_ptr<xmlNode> to_xml(const lhe &doc);
    std::shared_ptr<xmlNode> load_xml(const std::string &filename);

    /// Minimal SLHA container: BLOCK entries with integer indices -> double values,
    /// and DECAY widths keyed by PDG id. Comments are discarded.
    class slha
    {
    public:
        void read(std::istream &in);
        static slha parse(std::istream &in)
        {
            slha s;
            s.read(in);
            return s;
        }

        void write(std::ostream &out = std::cout,
                   int value_precision = 6,
                   bool scientific = true,
                   const std::string &indent = "      ") const;

        // Get with fallback
        double get(const std::string &block,
                   std::initializer_list<int> indices,
                   double fallback = 0.0) const;
        double get(const std::string &block,
                   int index,
                   double fallback = 0.0) const;

        // Set
        void set(const std::string &block,
                 std::initializer_list<int> indices,
                 double value);
        void set(const std::string &block,
                 int index,
                 double value);

        // -------- DECAY: Get / Set --------
        // With fallback
        double get_decay(int pid, double fallback = 0.0) const;
        void set_decay(int pid, double width);

        // Introspection
        bool has_block(const std::string &block) const;
        bool has_entry(const std::string &block, std::initializer_list<int> indices) const;

    private:
        struct VecLess
        {
            bool operator()(const std::vector<int> &a, const std::vector<int> &b) const
            {
                return a < b; // lexicographic
            }
        };
        struct BlockData
        {
            std::map<std::vector<int>, double, VecLess> entries;
        };

        std::map<std::string, BlockData> blocks_; // UPPER block -> data
        std::map<int, double> decays_;            // pid -> width

        static std::string trim(const std::string &s);
        static bool starts_with_ci(const std::string &s, const char *prefix);
        static std::string upper(std::string s);

        static std::string indices_to_string(std::initializer_list<int> indices);
    };

    slha to_slha(const std::string &slha_text);
    slha to_slha(std::istream &slha_stream);
    slha to_slha(std::shared_ptr<xmlNode> node);
    slha to_slha(const lhe &doc);
    slha load_slha(const std::string &filename);

} // namespace REX

#endif // DEFINING _REX_H_ FUNCTIONALITY

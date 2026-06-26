#pragma once

#include <cstdio>
#include <format>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace madspace {

template <class... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

template <typename T>
using nested_vector2 = std::vector<std::vector<T>>;
template <typename T>
using nested_vector3 = std::vector<std::vector<std::vector<T>>>;
template <typename T>
using nested_vector4 = std::vector<std::vector<std::vector<std::vector<T>>>>;

// Unfortunately nvcc does not support C++23 yet, so we implement our own zip function
// here (based on https://github.com/alemuntoni/zip-views), otherwise use the standard
// library function

namespace detail {

inline void print_impl(
    std::FILE* stream, bool new_line, std::string_view fmt, std::format_args args
) {
    std::string str = std::vformat(fmt, args);
    if (new_line) {
        str.push_back('\n');
    }
    fwrite(str.data(), 1, str.size(), stream);
}

template <typename... Args, std::size_t... Index>
bool any_match_impl(
    const std::tuple<Args...>& lhs,
    const std::tuple<Args...>& rhs,
    std::index_sequence<Index...>
) {
    auto result = false;
    result = (... || (std::get<Index>(lhs) == std::get<Index>(rhs)));
    return result;
}

template <typename... Args>
bool any_match(const std::tuple<Args...>& lhs, const std::tuple<Args...>& rhs) {
    return any_match_impl(lhs, rhs, std::index_sequence_for<Args...>{});
}

template <std::ranges::viewable_range... Rng>
class zip_iterator {
public:
    using value_type = std::tuple<std::ranges::range_reference_t<Rng>...>;

    zip_iterator() = delete;
    zip_iterator(std::ranges::iterator_t<Rng>&&... iters) :
        _iters{std::forward<std::ranges::iterator_t<Rng>>(iters)...} {}

    zip_iterator& operator++() {
        std::apply([](auto&&... args) { ((++args), ...); }, _iters);
        return *this;
    }

    zip_iterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    bool operator!=(const zip_iterator& other) const { return !(*this == other); }

    bool operator==(const zip_iterator& other) const {
        return any_match(_iters, other._iters);
    }

    value_type operator*() {
        return std::apply([](auto&&... args) { return value_type(*args...); }, _iters);
    }

private:
    std::tuple<std::ranges::iterator_t<Rng>...> _iters;
};

template <std::ranges::viewable_range... T>
class zipper {
public:
    using zip_type = zip_iterator<T...>;

    template <typename... Args>
    zipper(Args&&... args) : _args{std::forward<Args>(args)...} {}

    zip_type begin() {
        return std::apply(
            [](auto&&... args) { return zip_type(std::ranges::begin(args)...); }, _args
        );
    }
    zip_type end() {
        return std::apply(
            [](auto&&... args) { return zip_type(std::ranges::end(args)...); }, _args
        );
    }

private:
    std::tuple<T...> _args;
};

} // namespace detail

template <std::ranges::viewable_range... T>
auto zip(T&&... t) {
    return detail::zipper<T...>{std::forward<T>(t)...};
}

template <typename... Args>
inline void print(std::format_string<Args...> fmt, Args&&... args) {
    detail::print_impl(stdout, false, fmt.get(), std::make_format_args(args...));
}

template <typename... Args>
inline void print(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args) {
    detail::print_impl(stream, false, fmt.get(), std::make_format_args(args...));
}

template <typename... Args>
inline void println(std::format_string<Args...> fmt, Args&&... args) {
    detail::print_impl(stdout, true, fmt.get(), std::make_format_args(args...));
}

template <typename... Args>
inline void
println(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args) {
    detail::print_impl(stream, true, fmt.get(), std::make_format_args(args...));
}

template <typename T>
class NamedVector {
public:
    NamedVector() = default;
    NamedVector(const std::vector<std::string>& keys, const std::vector<T>& values) {
        if (keys.size() != values.size()) {
            throw std::invalid_argument("keys and values must have the same size");
        }
        reserve(values.size());
        for (auto [key, value] : zip(keys, values)) {
            push_back(key, value);
        }
    }
    NamedVector(const std::initializer_list<std::pair<std::string, T>>& items) {
        reserve(items.size());
        for (auto& [key, value] : items) {
            push_back(key, value);
        }
    }
    NamedVector(const std::vector<std::pair<std::string, T>>& items) {
        reserve(items.size());
        for (auto& [key, value] : items) {
            push_back(key, value);
        }
    }

    const std::vector<T>& values() const { return _values; }
    const std::unordered_map<std::string, std::size_t>& index_map() const {
        return _index_map;
    }
    std::vector<std::string> keys() const {
        std::vector<std::string> ret(size());
        for (auto& [key, index] : index_map()) {
            ret.at(index) = key;
        }
        return ret;
    }

    decltype(auto) begin() { return _values.begin(); }
    decltype(auto) begin() const { return _values.begin(); }
    decltype(auto) rbegin() { return _values.rbegin(); }
    decltype(auto) rbegin() const { return _values.rbegin(); }
    decltype(auto) end() { return _values.end(); }
    decltype(auto) end() const { return _values.end(); }
    decltype(auto) rend() { return _values.rend(); }
    decltype(auto) rend() const { return _values.rend(); }

    decltype(auto) front() { return _values.at(0); }
    decltype(auto) front() const { return _values.at(0); }
    decltype(auto) back() { return _values.at(size() - 1); }
    decltype(auto) back() const { return _values.at(size() - 1); }

    T& at(std::size_t index) { return _values.at(index); }
    const T& at(std::size_t index) const { return _values.at(index); }
    T& at(const std::string& key) { return _values.at(_index_map.at(key)); }
    const T& at(const std::string& key) const { return _values.at(_index_map.at(key)); }
    T& operator[](std::size_t index) { return at(index); }
    const T& operator[](std::size_t index) const { return at(index); }
    T& operator[](const std::string& key) { return at(key); }
    const T& operator[](const std::string& key) const { return at(key); }

    bool empty() const { return _values.empty(); }
    std::size_t size() const { return _values.size(); }
    void reserve(std::size_t capacity) { _values.reserve(capacity); }

    void push_back(const std::string& key, const T& value) {
        if (_index_map.contains(key)) {
            throw std::invalid_argument("Key already present in NamedVector");
        }
        _index_map[key] = _values.size();
        _values.push_back(value);
    }
    void insert_back(const NamedVector<T>& other) {
        std::size_t old_size = size();
        _values.insert(_values.end(), other.values().begin(), other.values().end());
        for (auto& [key, index] : other.index_map()) {
            if (_index_map.contains(key)) {
                throw std::invalid_argument("Key already present in NamedVector");
            }
            _index_map[key] = index + old_size;
        }
    }
    NamedVector<T>
    sort_like(const std::unordered_map<std::string, std::size_t>& index_map) const {
        if (index_map.size() != size()) {
            throw std::invalid_argument("index map must be the same size");
        }
        NamedVector<T> ret;
        ret._values.resize(size());
        ret._index_map = index_map;
        for (auto& [key, index] : index_map) {
            ret._values.at(index) = at(key);
        }
        return ret;
    }

private:
    std::vector<T> _values;
    std::unordered_map<std::string, std::size_t> _index_map;
};

} // namespace madspace

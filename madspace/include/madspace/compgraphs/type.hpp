#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace madspace {

/**
 * Element data type of a compute-graph @ref Value.
 *
 * `dt_int32`, `dt_float32` and `dt_float64` are ordinary integer and
 * floating-point tensors.
 * `batch_sizes` marks a value that instead carries a list of batch sizes, used
 * to split and recombine batches in multi-channel functions.
 */
enum class DataType {
    dt_int32,    ///< 32-bit integer tensor.
    dt_float32,  ///< Single-precision floating-point tensor.
    dt_float64,  ///< Double-precision floating-point tensor.
    batch_sizes, ///< A list of batch sizes rather than tensor data.
};

using me_int_t = int;

template <typename T>
concept ScalarType =
    std::same_as<T, me_int_t> || std::same_as<T, float> || std::same_as<T, double>;

/// Scalar types that can be stored in a literal @ref Value.
template <typename T>
concept LiteralType = std::same_as<T, me_int_t> || std::same_as<T, double>;

/// The `DataType` corresponding to the C++ element type `T`.
template <ScalarType T>
constexpr DataType dtype_of() {
    if constexpr (std::is_same_v<T, me_int_t>) {
        return DataType::dt_int32;
    } else if constexpr (std::is_same_v<T, float>) {
        return DataType::dt_float32;
    } else {
        return DataType::dt_float64;
    }
}

/// True for the floating-point data types `dt_float32` and `dt_float64`.
inline bool is_float(DataType dtype) {
    return dtype == DataType::dt_float32 || dtype == DataType::dt_float64;
}

/**
 * Symbolic size of the batch dimension of a @ref Value.
 *
 * A batch size is one of: the scalar size `one` (no batch dimension), a named
 * symbol such as `"batch_size"`, an anonymous symbol created for an
 * intermediate result, or a sum of such terms with integer coefficients.
 * Sizes are added and compared symbolically, so a graph can be type checked
 * without knowing the actual number of events.
 */
class BatchSize {
public:
    using Named = std::string;
    class UnnamedBody {
    public:
        UnnamedBody() : id(counter++) {}
        friend std::ostream& operator<<(std::ostream& out, const BatchSize& batch_size);
        friend void to_json(nlohmann::json& j, const BatchSize& batch_size);
        bool operator==(const UnnamedBody& other) const { return id == other.id; }
        bool operator!=(const UnnamedBody& other) const { return id != other.id; }

    private:
        static std::size_t counter;
        std::size_t id;
    };
    using Unnamed = std::shared_ptr<UnnamedBody>;
    using One = std::monostate;
    using Compound = std::map<std::variant<Named, Unnamed, One>, int>;

    static const BatchSize zero; ///< The empty batch size (a sum of no terms).
    static const BatchSize one;  ///< The scalar batch size (no batch dimension).

    /// Named batch size such as `"batch_size"`.
    /// @param name symbol name
    BatchSize(const std::string& name) : value(name) {}
    /// Scalar batch size (no batch dimension).
    BatchSize(One value) : value(value) {}
    /// Fresh anonymous batch size, distinct from every other.
    BatchSize() : value(std::make_shared<UnnamedBody>()) {}
    /// Symbolic sum of two batch sizes.
    BatchSize operator+(const BatchSize& other) const { return add(other, 1); }
    /// Symbolic difference of two batch sizes.
    BatchSize operator-(const BatchSize& other) const { return add(other, -1); }
    bool operator==(const BatchSize& other) const { return value == other.value; }
    bool operator!=(const BatchSize& other) const { return value != other.value; }
    /// Human-readable representation such as `"batch_size + 1"`.
    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream& out, const BatchSize& batch_size);
    friend void to_json(nlohmann::json& j, const BatchSize& batch_size);
    friend void from_json(const nlohmann::json& j, BatchSize& batch_size);

private:
    BatchSize(Compound value) : value(value) {}
    BatchSize(Unnamed value) : value(value) {}
    BatchSize add(const BatchSize& other, int factor) const;

    std::variant<Named, Unnamed, One, Compound> value;
};

void to_json(nlohmann::json& j, const BatchSize& batch_size);
void to_json(nlohmann::json& j, const BatchSize& batch_size);
void from_json(const nlohmann::json& j, BatchSize& batch_size);

/**
 * Full type of a compute-graph @ref Value. It combines an element `DataType`, a
 * symbolic @ref BatchSize and a static trailing shape.
 *
 * The logical tensor shape is `(batch_size, shape...)`. A `batch_sizes` value
 * instead holds @p batch_size_list and has no element shape.
 */
struct Type {
    /// Element data type.
    DataType dtype;
    /// Symbolic size of the leading batch dimension.
    BatchSize batch_size;
    /// Static trailing dimensions, without the batch dimension.
    std::vector<int> shape;
    /// For a `batch_sizes` value: the list of sizes it carries.
    std::vector<BatchSize> batch_size_list;

    /**
     * @param dtype element data type
     * @param batch_size symbolic size of the batch dimension
     * @param shape static trailing dimensions
     */
    Type(DataType dtype, BatchSize batch_size, const std::vector<int>& shape) :
        dtype(dtype), batch_size(batch_size), shape(shape) {}
    /// Construct a `batch_sizes` type carrying @p batch_size_list.
    Type(const std::vector<BatchSize>& batch_size_list) :
        dtype(DataType::batch_sizes),
        batch_size(BatchSize::one),
        batch_size_list(batch_size_list) {}
};

std::ostream& operator<<(std::ostream& out, const BatchSize& batch_size);
std::ostream& operator<<(std::ostream& out, const DataType& dtype);
std::ostream& operator<<(std::ostream& out, const Type& type);

inline bool operator==(const Type& lhs, const Type& rhs) {
    return lhs.dtype == rhs.dtype && lhs.batch_size == rhs.batch_size &&
        lhs.shape == rhs.shape;
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
    return lhs.dtype != rhs.dtype || lhs.batch_size != rhs.batch_size ||
        lhs.shape != rhs.shape;
}

using TypeVec = std::vector<Type>;

const Type single_float{DataType::dt_float64, BatchSize::One{}, {}};
const Type single_int{DataType::dt_int32, BatchSize::One{}, {}};
inline Type single_float_array(int count) {
    return {DataType::dt_float64, BatchSize::one, {count}};
}
inline Type single_int_array(int count) {
    return {DataType::dt_int32, BatchSize::one, {count}};
}
inline Type single_float_array_2d(int count1, int count2) {
    return {DataType::dt_float64, BatchSize::one, {count1, count2}};
}
inline Type single_int_array_2d(int count1, int count2) {
    return {DataType::dt_int32, BatchSize::one, {count1, count2}};
}

const BatchSize batch_size = BatchSize("batch_size");
Type batch_size_array(int count);
/// A `batch_sizes` `Type` splitting `batch_size` into `count` symbolic
/// per-channel sizes that add up to the total.
Type multichannel_batch_size(int count);
const Type batch_float{DataType::dt_float64, batch_size, {}};
const Type batch_int{DataType::dt_int32, batch_size, {}};
const Type batch_four_vec{DataType::dt_float64, batch_size, {4}};
/// A `float` `Type` with a batch dimension and a static trailing shape of
/// `count` elements.
inline Type batch_float_array(int count) {
    return {DataType::dt_float64, batch_size, {count}};
}
inline Type batch_int_array(int count) {
    return {DataType::dt_int32, batch_size, {count}};
}
/// A `Type` of `count` four-vectors: batch dimension plus a trailing shape of
/// `(count, 4)`.
inline Type batch_four_vec_array(int count) {
    return {DataType::dt_float64, batch_size, {count, 4}};
}

using TensorValue = std::tuple<
    std::vector<int>,
    std::variant<std::vector<me_int_t>, std::vector<double>>>; // TODO: make this a
                                                               // class

using LiteralValue = std::variant<me_int_t, double, TensorValue, std::monostate>;

/**
 * A node in a compute graph: a typed tensor that is either a literal constant
 * or a reference to a graph local.
 *
 * @ref FunctionBuilder methods take and return `Value`s. A value with
 * `local_index >= 0` refers to the output of an earlier instruction. Otherwise
 * @p literal_value holds an inline scalar or dense tensor constant. Scalars and
 * nested `std::vector`s convert to a `Value` implicitly.
 */
struct Value {
    /// Type of the value.
    Type type;
    /// Inline constant, or `std::monostate` for a graph local.
    LiteralValue literal_value;
    /// Index of the referenced graph local, or -1 for a literal.
    int local_index = -1;

    /// Empty value (no literal, no local); tests as `false`.
    Value() : type(single_float), literal_value(std::monostate{}) {}

    /// Scalar integer constant.
    Value(me_int_t value) : type(single_int), literal_value(value) {}
    /// Scalar floating-point constant.
    Value(double value) : type(single_float), literal_value(value) {}

    template <LiteralType T>
    Value(const std::vector<std::vector<T>>& values) :
        Value(
            [&] {
                std::size_t outer_size = values.size();
                std::size_t inner_size = values.at(0).size();
                std::vector<T> flat_values;
                for (auto& vec : values) {
                    if (vec.size() != inner_size) {
                        throw std::invalid_argument(
                            "All inner vectors must have the same size"
                        );
                    }
                }
                for (std::size_t j = 0; j < inner_size; ++j) {
                    for (std::size_t i = 0; i < outer_size; ++i) {
                        flat_values.push_back(values.at(i).at(j));
                    }
                }
                return flat_values;
            }(),
            {static_cast<int>(values.size()), static_cast<int>(values.at(0).size())}
        ) {}

    template <LiteralType T>
    Value(const std::vector<T>& values, const std::vector<int>& shape = {}) :
        type{
            dtype_of<T>(),
            BatchSize::one,
            shape.size() == 0 ? std::vector<int>{static_cast<int>(values.size())}
                              : shape
        },
        literal_value(TensorValue(type.shape, values)) {
        std::size_t prod = 1;
        for (auto size : type.shape) {
            prod *= size;
        }
        if (prod != values.size()) {
            throw std::invalid_argument(
                "size of value vector not compatible with given shape"
            );
        }
    }

    Value(Type _type, int _local_index) :
        type(_type), literal_value(std::monostate{}), local_index(_local_index) {}
    Value(Type _type, LiteralValue _literal_value, int _local_index = -1) :
        type(_type), literal_value(_literal_value), local_index(_local_index) {}

    operator bool() {
        return !(
            local_index == -1 && std::holds_alternative<std::monostate>(literal_value)
        );
    }
};

using ValueVec = std::vector<Value>;

void to_json(nlohmann::json& j, const DataType& dtype);
void to_json(nlohmann::json& j, const Value& value);
void from_json(const nlohmann::json& j, DataType& dtype);
void from_json(const nlohmann::json& j, Value& dtype);

} // namespace madspace

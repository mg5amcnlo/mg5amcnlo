#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace madspace {

enum class DataType { dt_int, dt_float, batch_sizes };

using me_int_t = int;

template <typename T>
concept ScalarType = std::same_as<T, me_int_t> || std::same_as<T, double>;

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

    static const BatchSize zero;
    static const BatchSize one;

    BatchSize(const std::string& name) : value(name) {}
    BatchSize(One value) : value(value) {}
    BatchSize() : value(std::make_shared<UnnamedBody>()) {}
    BatchSize operator+(const BatchSize& other) const { return add(other, 1); }
    BatchSize operator-(const BatchSize& other) const { return add(other, -1); }
    bool operator==(const BatchSize& other) const { return value == other.value; }
    bool operator!=(const BatchSize& other) const { return value != other.value; }
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

struct Type {
    DataType dtype;
    BatchSize batch_size;
    std::vector<int> shape;
    std::vector<BatchSize> batch_size_list;

    Type(DataType dtype, BatchSize batch_size, const std::vector<int>& shape) :
        dtype(dtype), batch_size(batch_size), shape(shape) {}
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

const Type single_float{DataType::dt_float, BatchSize::One{}, {}};
const Type single_int{DataType::dt_int, BatchSize::One{}, {}};
inline Type single_float_array(int count) {
    return {DataType::dt_float, BatchSize::one, {count}};
}
inline Type single_int_array(int count) {
    return {DataType::dt_int, BatchSize::one, {count}};
}
inline Type single_float_array_2d(int count1, int count2) {
    return {DataType::dt_float, BatchSize::one, {count1, count2}};
}
inline Type single_int_array_2d(int count1, int count2) {
    return {DataType::dt_int, BatchSize::one, {count1, count2}};
}

const BatchSize batch_size = BatchSize("batch_size");
Type batch_size_array(int count);
Type multichannel_batch_size(int count);
const Type batch_float{DataType::dt_float, batch_size, {}};
const Type batch_int{DataType::dt_int, batch_size, {}};
const Type batch_four_vec{DataType::dt_float, batch_size, {4}};
inline Type batch_float_array(int count) {
    return {DataType::dt_float, batch_size, {count}};
}
inline Type batch_four_vec_array(int count) {
    return {DataType::dt_float, batch_size, {count, 4}};
}

using TensorValue = std::tuple<
    std::vector<int>,
    std::variant<std::vector<me_int_t>, std::vector<double>>>; // TODO: make this a
                                                               // class

using LiteralValue = std::variant<me_int_t, double, TensorValue, std::monostate>;

struct Value {
    Type type;
    LiteralValue literal_value;
    int local_index = -1;

    Value() : type(single_float), literal_value(std::monostate{}) {}

    Value(me_int_t value) : type(single_int), literal_value(value) {}
    Value(double value) : type(single_float), literal_value(value) {}

    template <ScalarType T>
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

    template <ScalarType T>
    Value(const std::vector<T>& values, const std::vector<int>& shape = {}) :
        type{
            std::is_same_v<T, me_int_t> ? DataType::dt_int : DataType::dt_float,
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

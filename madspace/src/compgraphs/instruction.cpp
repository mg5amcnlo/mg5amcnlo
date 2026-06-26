#include "madspace/compgraphs/instruction.hpp"
#include "madspace/umami.h"

#include <algorithm>
#include <format>
#include <ranges>
#include <sstream>
#include <tuple>

using namespace madspace;

ShapeExpr::ShapeExpr(const char* expr) {
    int state = 0;
    int sign = 1;
    int factor = 0;
    bool no_digits = true;
    for (int i = 0; expr[i] != '\0'; ++i) {
        char c = expr[i];
        int digit = -1;
        int sign_value = 0;
        bool is_var = false;
        if (c == ' ') {
            continue;
        } else if (c == '-') {
            sign_value = -1;
        } else if (c == '+') {
            sign_value = 1;
        } else if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'z') {
            is_var = true;
        } else {
            throw std::invalid_argument(
                std::format("Invalid character {} in size expression", c)
            );
        }
        switch (state) {
        case 0: // first character
            if (sign_value != 0) {
                sign = sign_value;
                state = 1;
            } else if (digit != -1) {
                factor = digit;
                no_digits = false;
                state = 1;
            } else if (is_var) {
                terms.emplace_back(c, 1);
                state = 2;
            }
            break;
        case 1: // expect sign, digit or variable
            if (sign_value != 0) {
                if (no_digits) {
                    sign *= sign_value;
                } else {
                    terms.emplace_back(0, sign * factor);
                    sign = 1;
                    factor = 0;
                    no_digits = true;
                    state = 2;
                }
            } else if (digit != -1) {
                factor = 10 * factor + digit;
                no_digits = false;
            } else if (is_var) {
                terms.emplace_back(c, no_digits ? sign : sign * factor);
                sign = 1;
                factor = 0;
                no_digits = true;
                state = 2;
            }
            break;
        case 2: // expect sign
            if (sign_value != 0) {
                sign *= sign_value;
                state = 1;
            } else {
                throw std::invalid_argument("Invalid size expression");
            }
            break;
        }
    }
    if (!no_digits) {
        terms.emplace_back(0, sign * factor);
    }
}

bool ShapeExpr::check_and_update(std::map<char, int>& variables, int value) const {
    char unknown_var = 0;
    int unknown_factor = 0;
    int offset = 0;
    for (auto& [var_name, factor] : terms) {
        if (var_name == 0) {
            offset += factor;
        } else if (auto search = variables.find(var_name); search != variables.end()) {
            offset += factor * search->second;
        } else if (unknown_var == 0) {
            unknown_var = var_name;
            unknown_factor = factor;
        } else {
            return false;
        }
    }
    if (unknown_var == 0) {
        return value == offset;
    }
    if ((value - offset) % unknown_factor != 0) {
        return false;
    }
    variables[unknown_var] = (value - offset) / unknown_factor;
    return true;
}

std::optional<int> ShapeExpr::evaluate(const std::map<char, int>& variables) const {
    int value = 0;
    for (auto& [var_name, factor] : terms) {
        if (var_name == 0) {
            value += factor;
        } else if (auto search = variables.find(var_name); search != variables.end()) {
            value += factor * search->second;
        } else {
            return std::nullopt;
        }
    }
    return value;
}

void Instruction::check_arg_count(const ValueVec& args, std::size_t count) const {
    if (args.size() != count) {
        throw std::invalid_argument(
            std::format("{}: expected {} arguments, got {}", name(), count, args.size())
        );
    }
}

me_int_t Instruction::int_literal_arg(
    const ValueVec& args, std::size_t index, bool check_non_negative
) const {
    auto& arg_val = args.at(index);
    auto& arg_type = arg_val.type;
    if (arg_type.dtype != DataType::dt_int || arg_type.batch_size != BatchSize::one ||
        arg_type.shape.size() != 0 ||
        !std::holds_alternative<me_int_t>(arg_val.literal_value)) {
        throw std::invalid_argument(
            std::format("{}, argument {}: expected integer constant", name(), index + 1)
        );
    }
    me_int_t value = std::get<me_int_t>(arg_val.literal_value);
    if (check_non_negative && value < 0) {
        throw std::invalid_argument(
            std::format("{}, argument {}: must be positive", name(), index + 1)
        );
    }
    return value;
}

TypeVec SimpleInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, inputs.size());
    std::map<char, int> variables;
    std::vector<int> wildcard_shape;
    bool found_wildcard(false);
    BatchSize batch_size = BatchSize::one;

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& arg = args.at(i);
        auto& [arg_dtype, arg_batch_size, arg_shape, _] = arg.type;
        auto& [input_dtype, is_single, input_shape, is_size] = inputs.at(i);

        if (is_size) {
            me_int_t int_val = int_literal_arg(args, i);
            char var_name = std::get<ShapeExpr>(input_shape.at(0)).first_var_name();
            if (variables.find(var_name) != variables.end()) {
                throw std::invalid_argument(
                    std::format("{}, argument {}: size already defined", name(), i + 1)
                );
            }
            variables[var_name] = int_val;
            continue;
        }

        if (arg_dtype == DataType::batch_sizes) {
            throw std::invalid_argument(
                std::format(
                    "{}, argument {}: batch size list not accepted as argument",
                    name(),
                    i + 1
                )
            );
        }

        if (input_dtype != arg_dtype) {
            std::ostringstream expected_dt, got_dt;
            expected_dt << input_dtype;
            got_dt << arg_dtype;
            throw std::invalid_argument(
                std::format(
                    "{}, argument {}: expected dtype {}, got {}",
                    name(),
                    i + 1,
                    expected_dt.str(),
                    got_dt.str()
                )
            );
        }

        if (is_single) {
            if (arg_batch_size != BatchSize::one) {
                throw std::invalid_argument(
                    std::format(
                        "{}, argument {}: cannot have batch dimension", name(), i + 1
                    )
                );
            }
        } else {
            if (batch_size == BatchSize::one) {
                batch_size = arg_batch_size;
            } else if (arg_batch_size != BatchSize::one &&
                       batch_size != arg_batch_size) {
                throw std::invalid_argument(
                    std::format(
                        "{}, argument {}: incompatible batch size", name(), i + 1
                    )
                );
            }
        }

        auto wildcard_pos = std::find_if(
            input_shape.begin(), input_shape.end(), [](const auto& input_item) {
                return std::holds_alternative<std::monostate>(input_item);
            }
        );
        auto mod_input_shape = input_shape;
        if (wildcard_pos != input_shape.end()) {
            auto wildcard_index = wildcard_pos - input_shape.begin();
            if (!found_wildcard) {
                if (arg_shape.size() < input_shape.size() - 1) {
                    throw std::invalid_argument(
                        std::format(
                            "{}, argument {}: expected dimension of at least {}, got "
                            "{}",
                            name(),
                            i + 1,
                            input_shape.size() - 1,
                            arg_shape.size()
                        )
                    );
                }
                auto begin_pos = arg_shape.begin() + wildcard_index;
                auto end_pos = arg_shape.end() - (input_shape.end() - wildcard_pos) + 1;
                wildcard_shape.insert(wildcard_shape.begin(), begin_pos, end_pos);
                found_wildcard = true;
            }
            auto insert_pos =
                mod_input_shape.erase(mod_input_shape.begin() + wildcard_index);
            mod_input_shape.insert(
                insert_pos, wildcard_shape.begin(), wildcard_shape.end()
            );
        }

        if (arg_shape.size() != mod_input_shape.size()) {
            throw std::invalid_argument(
                std::format(
                    "{}, argument {}: expected dimension {}, got {}",
                    name(),
                    i + 1,
                    mod_input_shape.size(),
                    arg_shape.size()
                )
            );
        }
        for (size_t j = 0; j < arg_shape.size(); ++j) {
            auto input_item = mod_input_shape[j];
            auto arg_item = arg_shape[j];
            if (const auto shape_int = std::get_if<int>(&input_item)) {
                if (arg_item != *shape_int) {
                    throw std::invalid_argument(
                        std::format(
                            "{}, argument {}, dimension {}: expected size {}, got {}",
                            name(),
                            i + 1,
                            j,
                            *shape_int,
                            arg_item
                        )
                    );
                }
            } else {
                if (!std::get<ShapeExpr>(input_item)
                         .check_and_update(variables, arg_item)) {
                    throw std::invalid_argument(
                        std::format(
                            "{}, argument {}, dimension {}: incompatible size",
                            name(),
                            i + 1,
                            j
                        )
                    );
                }
            }
        }
    }

    TypeVec output_types;
    for (auto& [out_dtype, is_single, out_dyn_shape, is_size] : outputs) {
        std::vector<int> out_shape;
        for (auto& shape_item : out_dyn_shape) {
            if (const auto shape_int = std::get_if<int>(&shape_item)) {
                out_shape.push_back(*shape_int);
            } else {
                if (std::holds_alternative<std::monostate>(shape_item)) {
                    if (!found_wildcard) {
                        throw std::invalid_argument(
                            "Wildcard found in output signature, but not in input"
                        );
                    }
                    out_shape.insert(
                        out_shape.end(), wildcard_shape.begin(), wildcard_shape.end()
                    );
                } else {
                    auto value = std::get<ShapeExpr>(shape_item).evaluate(variables);
                    if (!value) {
                        throw std::invalid_argument(
                            "Output size could not be determined"
                        );
                    }
                    out_shape.push_back(*value);
                }
            }
        }
        output_types.push_back(
            Type{out_dtype, is_single ? BatchSize::one : batch_size, out_shape}
        );
    }
    return output_types;
}

TypeVec StackInstruction::signature(const ValueVec& args) const {
    if (args.size() == 0) {
        throw std::invalid_argument(
            "stack has to be called with at least one argument"
        );
    }
    auto type = args.at(0).type;
    BatchSize batch_size = BatchSize::one;
    std::size_t i = 1;
    for (auto& arg : args) {
        if (arg.type.dtype == DataType::batch_sizes) {
            throw std::invalid_argument(
                std::format(
                    "stack, argument {}: Batch size list not accepted as argument", i
                )
            );
        }
        if (batch_size == BatchSize::one) {
            batch_size = arg.type.batch_size;
        } else if (arg.type.batch_size != BatchSize::one &&
                   batch_size != arg.type.batch_size) {
            throw std::invalid_argument(
                std::format("stack, argument {}: incompatible batch size", i)
            );
        }
        if (arg.type.dtype != type.dtype || arg.type.shape != type.shape) {
            throw std::invalid_argument(
                "stack: all arguments must have the same shape and dtype"
            );
        }
        ++i;
    }
    int args_size = args.size();
    std::vector<int> out_shape{args_size};
    out_shape.insert(out_shape.end(), type.shape.begin(), type.shape.end());
    return {{type.dtype, batch_size, out_shape}};
}

TypeVec UnstackInstruction::signature(const ValueVec& args) const {
    if (args.size() != 1) {
        throw std::invalid_argument(
            std::format("unstack expects one argument, got {}", args.size())
        );
    }
    auto arg = args.at(0);
    if (arg.type.dtype == DataType::batch_sizes) {
        throw std::invalid_argument("Batch size list not accepted as argument");
    }
    if (arg.type.batch_size == BatchSize::one) {
        throw std::invalid_argument("Argument must have batch dimension");
    }
    if (arg.type.shape.size() == 0) {
        throw std::invalid_argument(
            "Argument of unstack must be at least one-dimensional"
        );
    }
    std::vector<int> out_shape(arg.type.shape.begin() + 1, arg.type.shape.end());
    return TypeVec(arg.type.shape[0], {arg.type.dtype, arg.type.batch_size, out_shape});
}

TypeVec StackSizesInstruction::signature(const ValueVec& args) const {
    std::vector<BatchSize> sizes;
    for (auto& arg : args) {
        if (arg.type.dtype != DataType::batch_sizes) {
            throw std::invalid_argument("Only batch size list accepted as argument");
        }
        sizes.push_back(arg.type.batch_size_list.at(0));
    }
    return {sizes};
}

TypeVec UnstackSizesInstruction::signature(const ValueVec& args) const {
    if (args.size() != 1) {
        throw std::invalid_argument(
            std::format("unstack_sizes expects one argument, got {}", args.size())
        );
    }
    auto arg = args.at(0);
    if (arg.type.dtype != DataType::batch_sizes) {
        throw std::invalid_argument("Only batch size list accepted as argument");
    }
    TypeVec out_types;
    for (auto& size : arg.type.batch_size_list) {
        out_types.push_back(std::vector<BatchSize>{size});
    }
    return out_types;
}

TypeVec BatchCatInstruction::signature(const ValueVec& args) const {
    if (args.size() == 0) {
        throw std::invalid_argument(
            "batch_cat has to be called with at least one argument"
        );
    }
    auto type = args.at(0).type;
    auto batch_size = BatchSize::zero;
    std::vector<BatchSize> arg_batch_sizes;
    for (auto& arg : args) {
        if (arg.type.dtype == DataType::batch_sizes) {
            throw std::invalid_argument("Batch size list not accepted as argument");
        }
        if (arg.type.batch_size == BatchSize::one) {
            throw std::invalid_argument("Argument must have batch dimension");
        }
        if (arg.type.dtype != type.dtype || arg.type.shape != type.shape) {
            throw std::invalid_argument(
                "All arguments must have the same shape and dtype"
            );
        }
        arg_batch_sizes.push_back(arg.type.batch_size);
        batch_size = batch_size + arg.type.batch_size;
    }
    int args_size = args.size();
    return {{type.dtype, batch_size, type.shape}, arg_batch_sizes};
}

TypeVec BatchSplitInstruction::signature(const ValueVec& args) const {
    if (args.size() != 2) {
        throw std::invalid_argument(
            std::format("batch_split expects two arguments, got {}", args.size())
        );
    }
    auto split_arg = args.at(0);
    if (split_arg.type.batch_size == BatchSize::one) {
        throw std::invalid_argument(
            "First argument of batch_split must have batch dimension"
        );
    }
    auto count_arg = args.at(1);
    if (count_arg.type.dtype != DataType::batch_sizes) {
        throw std::invalid_argument(
            "Second argument of batch_split must be batch size list"
        );
    }
    TypeVec out_types;
    auto last_batch_size = split_arg.type.batch_size;
    for (auto& batch_size : count_arg.type.batch_size_list) {
        if (&batch_size == &count_arg.type.batch_size_list.back()) {
            out_types.push_back(
                {split_arg.type.dtype, last_batch_size, split_arg.type.shape}
            );
        } else {
            out_types.push_back(
                {split_arg.type.dtype, batch_size, split_arg.type.shape}
            );
            last_batch_size = last_batch_size - batch_size;
        }
    }
    return out_types;
}

TypeVec CatInstruction::signature(const ValueVec& args) const {
    if (args.size() == 0) {
        throw std::invalid_argument("cat has to be called with at least one argument");
    }
    auto type = args.at(0).type;
    BatchSize batch_size = BatchSize::one;
    std::size_t cat_dim = 0;
    std::size_t i = 1;
    for (auto& arg : args) {
        if (arg.type.dtype == DataType::batch_sizes) {
            throw std::invalid_argument(
                std::format(
                    "cat, argument {}: batch size list not accepted as argument", i
                )
            );
        }
        if (arg.type.shape.size() == 0) {
            throw std::invalid_argument(
                std::format(
                    "cat, argument {}: arguments must be at least 1-dimensional", i
                )
            );
        }
        cat_dim += arg.type.shape.at(0);
        if (batch_size == BatchSize::one) {
            batch_size = arg.type.batch_size;
        } else if (arg.type.batch_size != BatchSize::one &&
                   batch_size != arg.type.batch_size) {
            throw std::invalid_argument(
                std::format("cat, argument {}: incompatible batch size", i)
            );
        }
        if (arg.type.dtype != type.dtype ||
            !std::equal(
                arg.type.shape.begin() + 1,
                arg.type.shape.end(),
                type.shape.begin() + 1,
                type.shape.end()
            )) {
            throw std::invalid_argument(
                "cat: all arguments must have the same shape and dtype"
            );
        }
        ++i;
    }
    std::vector<int> out_shape{static_cast<int>(cat_dim)};
    out_shape.insert(out_shape.end(), type.shape.begin() + 1, type.shape.end());
    return {{type.dtype, batch_size, out_shape}};
}

TypeVec BatchSizeInstruction::signature(const ValueVec& args) const {
    if (args.size() == 0) {
        throw std::invalid_argument(
            "batch_size has to be called with at least one argument"
        );
    }
    BatchSize batch_size = BatchSize::one;
    std::size_t cat_dim = 0;
    std::size_t i = 1;
    for (auto& arg : args) {
        if (arg.type.dtype == DataType::batch_sizes) {
            throw std::invalid_argument(
                std::format(
                    "batch_size, argument {}: batch size list not accepted as argument",
                    i
                )
            );
        }
        if (batch_size == BatchSize::one) {
            batch_size = arg.type.batch_size;
        } else if (arg.type.batch_size != BatchSize::one &&
                   batch_size != arg.type.batch_size) {
            throw std::invalid_argument(
                std::format("batch_size, argument {}: incompatible batch size", i)
            );
        }
        ++i;
    }
    return {{{batch_size}}};
}

TypeVec OffsetIndicesInstruction::signature(const ValueVec& args) const {
    if (args.size() != 2) {
        throw std::invalid_argument(
            std::format("offset_indices expects two arguments, got {}", args.size())
        );
    }
    auto batch_sizes_offset = args.at(0);
    auto batch_sizes_out = args.at(1);
    if (batch_sizes_offset.type.dtype != DataType::batch_sizes) {
        throw std::invalid_argument(
            "Argument 1 of offset_indices must be batch size list"
        );
    }
    if (batch_sizes_out.type.dtype != DataType::batch_sizes) {
        throw std::invalid_argument(
            "Argument 2 of offset_indices must be batch size list"
        );
    }
    if (batch_sizes_offset.type.batch_size_list.size() !=
        batch_sizes_out.type.batch_size_list.size()) {
        throw std::invalid_argument(
            "The two batch size lists must have the same length"
        );
    }
    BatchSize total_batch_size = std::accumulate(
        batch_sizes_out.type.batch_size_list.begin(),
        batch_sizes_out.type.batch_size_list.end(),
        BatchSize::zero
    );
    return {{DataType::dt_int, total_batch_size, {}}};
}

TypeVec FullInstruction::signature(const ValueVec& args) const {
    if (args.size() < 2) {
        throw std::invalid_argument("full expects at least two arguments");
    }

    auto& value_arg = args.at(0);
    if (value_arg.type.batch_size != BatchSize::one ||
        value_arg.type.shape.size() != 0 ||
        std::holds_alternative<std::monostate>(value_arg.literal_value)) {
        throw std::invalid_argument("full, argument 1: expected constant");
    }
    auto dtype = value_arg.type.dtype;

    auto& batch_size_arg = args.at(1);
    if (batch_size_arg.type.dtype != DataType::batch_sizes ||
        batch_size_arg.type.batch_size_list.size() != 1) {
        throw std::invalid_argument("full, argument 2: must be single batch size");
    }
    auto batch_size = batch_size_arg.type.batch_size_list.at(0);

    std::vector<int> shape;
    for (std::size_t i = 2; i < args.size(); ++i) {
        shape.push_back(int_literal_arg(args, i));
    }
    return {{dtype, batch_size, shape}};
}

TypeVec SqueezeInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 1);
    auto arg = args.at(0);
    if (arg.type.dtype == DataType::batch_sizes) {
        throw std::invalid_argument("Batch size list not accepted as argument");
    }
    if (arg.type.shape.size() == 0) {
        throw std::invalid_argument(
            "Argument of squeeze must be at least one-dimensional"
        );
    }
    std::vector<int> out_shape(arg.type.shape.begin() + 1, arg.type.shape.end());
    return {{arg.type.dtype, arg.type.batch_size, out_shape}};
}

TypeVec UnsqueezeInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 1);
    auto arg = args.at(0);
    if (arg.type.dtype == DataType::batch_sizes) {
        throw std::invalid_argument("Batch size list not accepted as argument");
    }
    std::vector<int> out_shape{1};
    out_shape.insert(out_shape.end(), arg.type.shape.begin(), arg.type.shape.end());
    return {{arg.type.dtype, arg.type.batch_size, out_shape}};
}

TypeVec AcceptNormInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 2);
    for (std::size_t i = 0; auto& arg : args) {
        if (arg.type.dtype == DataType::batch_sizes) {
            throw std::invalid_argument(
                std::format(
                    "accept_norm, argument {}: Batch size list not accepted as "
                    "argument",
                    i + 1
                )
            );
        }
        ++i;
    }
    return {{DataType::dt_float, BatchSize::one, {}}};
}

TypeVec RqsReshapeInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 2);
    int bin_count = int_literal_arg(args, 1);

    auto& input_type = args.at(0).type;
    if (input_type.dtype != DataType::dt_float || input_type.shape.size() != 1 ||
        input_type.shape.at(0) % (3 * bin_count + 1) != 0) {
        throw std::invalid_argument(
            std::format(
                "{}, argument 1: expected batch of n_dims * (3 * n_bins + 1) floats",
                name()
            )
        );
    }
    int dim = input_type.shape.at(0) / (3 * bin_count + 1);

    return {
        {DataType::dt_float, input_type.batch_size, {dim, bin_count}},
        {DataType::dt_float, input_type.batch_size, {dim, bin_count}},
        {DataType::dt_float, input_type.batch_size, {dim, bin_count + 1}},
    };
}

TypeVec NonzeroInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 1);
    auto& input_type = args.at(0).type;
    if (input_type.dtype != DataType::dt_float || input_type.shape.size() != 0) {
        throw std::invalid_argument(
            std::format("{}, argument 1: expected batch of floats", name())
        );
    }
    return {{DataType::dt_int, BatchSize(), {}}};
}

TypeVec BatchGatherInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 2);
    auto& indices_type = args.at(0).type;
    auto& values_type = args.at(1).type;
    if (indices_type.dtype != DataType::dt_int || indices_type.shape.size() != 0) {
        throw std::invalid_argument(
            std::format("{}, argument 1: expected batch of integers", name())
        );
    }
    if (values_type.dtype == DataType::batch_sizes) {
        throw std::invalid_argument(
            std::format("{}, argument 2: data type cannot be batch_sizes", name())
        );
    }
    return {{values_type.dtype, indices_type.batch_size, values_type.shape}};
}

TypeVec BatchScatterInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 3);
    auto& indices_type = args.at(0).type;
    auto& target_type = args.at(1).type;
    auto& source_type = args.at(2).type;
    if (indices_type.dtype != DataType::dt_int || indices_type.shape.size() != 0) {
        throw std::invalid_argument(
            std::format("{}, argument 1: expected batch of integers", name())
        );
    }
    if (target_type.dtype != DataType::dt_float &&
        target_type.dtype != DataType::dt_int) {
        throw std::invalid_argument(
            std::format("{}, argument 2: expected data type float or int", name())
        );
    }
    if (source_type.dtype != target_type.dtype ||
        source_type.batch_size != indices_type.batch_size ||
        source_type.shape != target_type.shape) {
        throw std::invalid_argument(
            std::format("{}, argument 3: incompatible source type", name())
        );
    }
    return {target_type};
}

TypeVec RandomInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 2);
    auto& batch_size_type = args.at(0).type;
    auto& count_arg = args.at(1);
    auto& count_type = count_arg.type;
    if (batch_size_type.dtype != DataType::batch_sizes ||
        batch_size_type.batch_size_list.size() != 1) {
        throw std::invalid_argument(
            std::format("{}, argument 1: expected single batch size", name())
        );
    }
    int count = int_literal_arg(args, 1);
    return {{DataType::dt_float, batch_size_type.batch_size_list.at(0), {count}}};
}

TypeVec RandomIntInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 2);
    auto& batch_size_type = args.at(0).type;
    auto& max_type = args.at(1).type;
    if (batch_size_type.dtype != DataType::batch_sizes ||
        batch_size_type.batch_size_list.size() != 1) {
        throw std::invalid_argument(
            std::format("{}, argument 1: expected single batch size", name())
        );
    }
    if (max_type.dtype != DataType::batch_sizes ||
        max_type.batch_size_list.size() != 1) {
        throw std::invalid_argument(
            std::format("{}, argument 2: expected single batch size", name())
        );
    }
    return {{DataType::dt_int, batch_size_type.batch_size_list.at(0), {}}};
}

TypeVec UnweightInstruction::signature(const ValueVec& args) const {
    check_arg_count(args, 2);
    auto& weights_type = args.at(0).type;
    auto& max_weight_type = args.at(1).type;
    if (weights_type.dtype != DataType::dt_float || weights_type.shape.size() != 0) {
        throw std::invalid_argument(
            std::format("{}, argument 1: expected batch of floats", name())
        );
    }
    if (max_weight_type.dtype != DataType::dt_float ||
        max_weight_type.batch_size != BatchSize::one ||
        max_weight_type.shape.size() != 0) {
        throw std::invalid_argument(
            std::format("{}, argument 2: expected single float", name())
        );
    }

    BatchSize out_batch_size;
    return {
        {DataType::dt_int, out_batch_size, {}}, {DataType::dt_float, out_batch_size, {}}
    };
}

TypeVec MatrixElementInstruction::signature(const ValueVec& args) const {
    std::size_t arg_count = args.size();
    if (arg_count < 3) {
        throw std::invalid_argument(
            "matrix_element: must have at least three arguments, specifying the "
            "matrix element index and the numbers of inputs and outputs"
        );
    }
    int_literal_arg(args, 0);
    std::size_t input_count = int_literal_arg(args, 1);
    std::size_t output_count = int_literal_arg(args, 2);
    std::size_t expected_arg_count = 3 + 2 * input_count + 2 * output_count;
    if (arg_count != expected_arg_count) {
        throw std::invalid_argument(
            std::format(
                "matrix_element: expected {} arguments, got {}",
                expected_arg_count,
                arg_count
            )
        );
    }
    BatchSize batch_size = BatchSize::one;
    for (std::size_t i = 0; i < input_count; ++i) {
        std::size_t input_key_index = 3 + 2 * i;
        std::size_t input_value_index = 3 + 2 * i + 1;
        int input_key = int_literal_arg(args, input_key_index);
        const Type& input_type = args.at(input_value_index).type;
        if (batch_size == BatchSize::one) {
            batch_size = input_type.batch_size;
        } else if (input_type.batch_size != BatchSize::one &&
                   batch_size != input_type.batch_size) {
            throw std::invalid_argument(
                std::format(
                    "matrix_element, argument {}: incompatible batch size", i + 1
                )
            );
        }
        switch (input_key) {
        case UMAMI_IN_MOMENTA:
            if (input_type.dtype != DataType::dt_float ||
                input_type.shape.size() != 2 || input_type.shape.at(1) != 4) {
                throw std::invalid_argument(
                    std::format(
                        "matrix_element, argument {}: expected array of four-momenta",
                        i + 1
                    )
                );
            }
            break;
        case UMAMI_IN_ALPHA_S:
        case UMAMI_IN_RANDOM_COLOR:
        case UMAMI_IN_RANDOM_HELICITY:
        case UMAMI_IN_RANDOM_DIAGRAM:
            if (input_type.dtype != DataType::dt_float ||
                input_type.shape.size() != 0) {
                throw std::invalid_argument(
                    std::format(
                        "matrix_element, argument {}: expected batch of floats", i + 1
                    )
                );
            }
            break;
        case UMAMI_IN_FLAVOR_INDEX:
        case UMAMI_IN_HELICITY_INDEX:
        case UMAMI_IN_DIAGRAM_INDEX:
        case UMAMI_IN_CHANNEL_INDEX:
            if (input_type.dtype != DataType::dt_int || input_type.shape.size() != 0) {
                throw std::invalid_argument(
                    std::format(
                        "matrix_element, argument {}: expected batch of integers", i + 1
                    )
                );
            }
            break;
        default:
            throw std::invalid_argument(
                std::format(
                    "matrix_element, argument {}: invalid input key", input_key_index
                )
            );
        }
    }
    TypeVec output_types;
    std::size_t output_offset = 3 + 2 * input_count;
    for (std::size_t i = 0; i < output_count; ++i) {
        std::size_t output_key_index = output_offset + 2 * i;
        std::size_t output_size_index = output_offset + 2 * i + 1;
        int output_key = int_literal_arg(args, output_key_index);
        int output_size = int_literal_arg(args, output_size_index);
        DataType dtype;
        std::vector<int> shape;
        switch (output_key) {
        case UMAMI_OUT_MATRIX_ELEMENT:
            dtype = DataType::dt_float;
            shape = {};
            break;
        case UMAMI_OUT_DIAGRAM_AMP2:
            dtype = DataType::dt_float;
            shape = {output_size};
            break;
        case UMAMI_OUT_COLOR_INDEX:
        case UMAMI_OUT_HELICITY_INDEX:
        case UMAMI_OUT_DIAGRAM_INDEX:
            dtype = DataType::dt_int;
            shape = {};
            break;
        default:
            throw std::invalid_argument(
                std::format("matrix_element, argument {}: invalid output key", i + 1)
            );
        }
        output_types.push_back({dtype, batch_size, shape});
    }
    return output_types;
}

const std::unordered_map<std::string, InstructionOwner>
madspace::build_instruction_set() {
#include "instruction_set_mixin.inc"
    std::unordered_map<std::string, InstructionOwner> instruction_set;
    for (auto& instruction : instructions) {
        instruction_set[instruction->name()] = std::move(instruction);
    }
    return instruction_set;
}

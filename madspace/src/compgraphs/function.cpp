#include "madspace/compgraphs/function.hpp"
#include "madspace/util.hpp"

#include <format>
#include <fstream>
#include <stdexcept>

using namespace madspace;
using json = nlohmann::json;

void Function::save(const std::string& file) const {
    std::ofstream f(file);
    json j;
    j = *this;
    f << j.dump();
}

Function Function::load(const std::string& file) {
    std::ifstream f(file);
    return json::parse(f).get<Function>();
}

std::ostream& madspace::operator<<(std::ostream& out, const Value& value) {
    std::visit(
        Overloaded{
            [&out](auto val) { out << val; },
            [&out](TensorValue val) {
                std::visit(
                    [&out](auto items) {
                        out << "{";
                        std::size_t i = 0;
                        for (auto item : items) {
                            if (i == items.size() - 1) {
                                out << item;
                            } else if (i == 20) {
                                out << "...";
                                break;
                            } else {
                                out << item << ", ";
                            }
                            ++i;
                        }
                        out << "}";
                    },
                    std::get<1>(val)
                );
            },
            [&out, value](std::monostate val) { out << "%" << value.local_index; }
        },
        value.literal_value
    );
    return out;
}

std::ostream& madspace::operator<<(std::ostream& out, const ValueVec& list) {
    bool first = true;
    for (auto value : list) {
        if (first) {
            first = false;
        } else {
            out << ", ";
        }
        out << value;
    }
    return out;
}

std::ostream& madspace::operator<<(std::ostream& out, const InstructionCall& call) {
    out << call.outputs << " = " << call.instruction->name();
    if (call.stream_index != 0) {
        out << "@" << call.stream_index;
    }
    out << "(" << call.inputs << ")";
    return out;
}

std::ostream& madspace::operator<<(std::ostream& out, const Function& func) {
    out << "Inputs:\n";
    for (auto [name, input] : zip(func.inputs().keys(), func.inputs())) {
        out << "  " << input << " : " << input.type << "=" << name << "\n";
    }
    out << "Globals:\n";
    for (auto& [name, global] : func.globals()) {
        out << "  " << global << " : " << global.type << " = " << name << "\n";
    }
    out << "Instructions:\n";
    for (auto& instr : func.instructions()) {
        out << "  " << instr << "\n";
    }
    out << "Outputs:\n";
    for (auto [name, output] : zip(func.outputs().keys(), func.outputs())) {
        out << "  " << name << "=" << output << " : " << output.type << "\n";
    }
    return out;
}

void madspace::to_json(json& j, const InstructionCall& call) {
    j = json{
        {"name", call.instruction->name()},
        {"inputs", call.inputs},
        {"outputs", call.outputs},
        {"stream_index", call.stream_index},
    };
}

void madspace::to_json(json& j, const Function& func) {
    json inputs(json::value_t::array);
    json outputs(json::value_t::array);
    json globals(json::value_t::array);
    for (auto [name, input] : zip(func.inputs().keys(), func.inputs())) {
        inputs.push_back(
            input.type.dtype == DataType::batch_sizes ?
            json{
                {"name", name},
                {"local", input},
                {"dtype", input.type.dtype},
                {"batch_sizes", input.type.batch_size_list},
            } : json{
                {"name", name},
                {"local", input},
                {"dtype", input.type.dtype},
                {"batch_size", input.type.batch_size},
                {"shape", input.type.shape},
            }
        );
    }
    for (auto [name, output] : zip(func.outputs().keys(), func.outputs())) {
        outputs.push_back(
            json{
                {"name", name},
                {"local", output},
                {"dtype", output.type.dtype},
                {"shape", output.type.shape},
            }
        );
    }
    for (auto& [name, global] : func.globals()) {
        globals.push_back(
            json{
                {"local", global},
                {"name", name},
                {"dtype", global.type.dtype},
                {"shape", global.type.shape},
            }
        );
    }
    j = json{
        {"inputs", inputs},
        {"outputs", outputs},
        {"globals", globals},
        {"instructions", func.instructions()},
    };
}

void madspace::from_json(const json& j, Function& func) {
    NamedVector<Type> input_types, output_types;
    std::vector<std::size_t> input_locals, output_locals;
    for (auto& j_input : j.at("inputs")) {
        auto name = j_input.at("name").get<std::string>();
        if (j_input.at("dtype").get<DataType>() == DataType::batch_sizes) {
            input_types.push_back(
                name, j_input.at("batch_sizes").get<std::vector<BatchSize>>()
            );
        } else {
            BatchSize batch_size = BatchSize::one;
            j_input.at("batch_size").get_to<BatchSize>(batch_size);
            input_types.push_back(
                name,
                Type(
                    j_input.at("dtype").get<DataType>(),
                    batch_size,
                    j_input.at("shape").get<std::vector<int>>()
                )
            );
        }
        input_locals.push_back(j_input.at("local").get<std::size_t>());
    }
    for (auto& j_output : j.at("outputs")) {
        output_types.push_back(
            j_output.at("name").get<std::string>(),
            Type(
                j_output.at("dtype").get<DataType>(),
                BatchSize::one,
                j_output.at("shape").get<std::vector<int>>()
            )
        );
        output_locals.push_back(j_output.at("local").get<std::size_t>());
    }

    FunctionBuilder fb(input_types, output_types);
    std::unordered_map<std::size_t, Value> locals;
    std::size_t i = 0;
    for (auto input_index : input_locals) {
        locals[input_index] = fb.input(i);
        ++i;
    }

    for (auto& j_global : j.at("globals")) {
        locals[j_global.at("local").get<std::size_t>()] = fb.global(
            j_global.at("name").get<std::string>(),
            j_global.at("dtype").get<DataType>(),
            j_global.at("shape").get<std::vector<int>>()
        );
    }

    for (auto& j_instr : j.at("instructions")) {
        ValueVec instr_inputs;
        for (auto& j_input : j_instr.at("inputs")) {
            instr_inputs.push_back(
                j_input.is_number_unsigned()
                    ? locals.at(j_input.get<std::size_t>())
                    : j_input.get<Value>()
            );
        }
        fb.set_current_stream(j_instr.at("stream_index").get<std::size_t>());
        auto instr_outputs =
            fb.instruction(j_instr.at("name").get<std::string>(), instr_inputs);
        for (auto [j_output, output] : zip(j_instr.at("outputs"), instr_outputs)) {
            locals[j_output.get<std::size_t>()] = output;
        }
    }

    i = 0;
    for (auto output_index : output_locals) {
        fb.output(i, locals[output_index]);
        ++i;
    }
    func = fb.function();
}

FunctionBuilder::FunctionBuilder(
    const NamedVector<Type>& input_types, const NamedVector<Type>& _output_types
) :
    _output_types(_output_types), _current_stream(0) {
    for (auto [name, input_type] : zip(input_types.keys(), input_types)) {
        Value value(input_type, _locals.size());
        _locals.push_back(value);
        _local_sources.push_back(-1);
        _inputs.push_back(name, value);
    }
    for (auto output_type : _output_types) {
        _outputs.push_back(std::nullopt);
    }
}

FunctionBuilder::FunctionBuilder(const Function& function) :
    _inputs(function.inputs()),
    _locals(function.inputs().values()),
    _current_stream(0) {
    TypeVec output_types;
    for (auto& output : function.outputs()) {
        output_types.push_back(output.type);
        _outputs.push_back(std::nullopt);
    }
}

ValueVec FunctionBuilder::instruction(const std::string& name, const ValueVec& args) {
    auto find_instr = instruction_set.find(name);
    if (find_instr == instruction_set.end()) {
        throw std::invalid_argument(std::format("Unknown instruction '{}'", name));
    }
    return instruction(find_instr->second.get(), args);
}

ValueVec
FunctionBuilder::instruction(InstructionPtr instruction, const ValueVec& args) {
    // check argument types and determine if all arguments are constant
    auto params = args;
    std::size_t arg_index = 0;
    std::size_t variable_args = 0;
    for (auto& arg : params) {
        ++arg_index;

        if (arg.local_index != -1) {
            if (arg.local_index < 0 || arg.local_index > _locals.size()) {
                throw std::invalid_argument(
                    std::format(
                        "{}, argument {}: inconsistent value (local index)",
                        instruction->name(),
                        arg_index
                    )
                );
            }
            auto local_value = _locals.at(arg.local_index);
            if (local_value.type != arg.type ||
                local_value.literal_value != arg.literal_value) {
                throw std::invalid_argument(
                    std::format(
                        "{}, argument {}: inconsistent value (type or value)",
                        instruction->name(),
                        arg_index
                    )
                );
            }
            ++variable_args;
        } else if (std::holds_alternative<std::monostate>(arg.literal_value)) {
            throw std::invalid_argument(
                std::format(
                    "{}, argument {}: undefined value", instruction->name(), arg_index
                )
            );
        } else if (std::holds_alternative<TensorValue>(arg.literal_value)) {
            ++variable_args;
        }
    }

    // perform simple arithmetic operations on constant arguments
    int opcode = instruction->opcode();
    if (variable_args == 0) {
        switch (opcode) {
        case opcodes::add: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {arg0 + arg1};
        }
        case opcodes::sub: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {arg0 - arg1};
        }
        case opcodes::mul: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {arg0 * arg1};
        }
        case opcodes::sqrt: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            return {std::sqrt(arg0)};
        }
        case opcodes::square: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            return {arg0 * arg0};
        }
        case opcodes::min: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {std::min(arg0, arg1)};
        }
        case opcodes::max: {
            double arg0 = std::get<double>(args.at(0).literal_value);
            double arg1 = std::get<double>(args.at(1).literal_value);
            return {std::max(arg0, arg1)};
        }
        }
    } else if (variable_args == 1 && args.size() == 2) {
        std::size_t var_index = args.at(0).local_index == -1;
        auto& var_arg = args.at(var_index);
        auto& const_arg = args.at(1 - var_index);
        switch (opcode) {
        case opcodes::add:
        case opcodes::sub:
            if (std::get<double>(const_arg.literal_value) == 0.) {
                return {var_arg};
            }
            break;
        case opcodes::mul:
            if (std::get<double>(const_arg.literal_value) == 1.) {
                return {var_arg};
            }
            break;
        }
    }

    // create local variables for constants
    std::vector<std::size_t> cache_key;
    cache_key.push_back(opcode);
    cache_key.push_back(_current_stream);
    for (auto& arg : params) {
        register_local(arg);
        cache_key.push_back(arg.local_index);
    }

    // simplify square(sqrt(x))
    if (opcode == opcodes::square) {
        int source_index = _local_sources.at(args.at(0).local_index);
        if (source_index != -1) {
            auto& source_instr = _instructions.at(source_index);
            if (source_instr.instruction->opcode() == opcodes::sqrt) {
                return {_locals.at(source_instr.inputs.at(0).local_index)};
            }
        }
    }

    // check for cached result (for deterministic instructions)
    if (opcode != opcodes::random && opcode != opcodes::unweight) {
        auto find_instr = _instruction_cache.find(cache_key);
        if (find_instr != _instruction_cache.end()) {
            ValueVec call_outputs;
            for (auto output_local_index : find_instr->second) {
                call_outputs.push_back(_locals.at(output_local_index));
            }
            return call_outputs;
        }
    }

    // generate instruction and create local variables for output
    auto output_types = instruction->signature(params);
    ValueVec call_outputs;
    std::vector<std::size_t> output_locals;
    for (const auto& type : output_types) {
        Value value(type, _locals.size());
        _locals.push_back(value);
        _local_sources.push_back(_instructions.size());
        call_outputs.push_back(value);
        output_locals.push_back(value.local_index);
    }
    _instructions.push_back(
        InstructionCall{instruction, params, call_outputs, _current_stream}
    );
    _instruction_use_count.push_back(0);
    _instruction_cache[cache_key] = output_locals;
    for (auto& param : params) {
        int param_source = _local_sources.at(param.local_index);
        if (param_source != -1) {
            ++_instruction_use_count.at(param_source);
        }
    }
    return call_outputs;
}

Function FunctionBuilder::function() {
    NamedVector<Value> func_outputs;

    auto output_names = _output_types.keys();
    for (std::size_t output_index = 0; auto output : _outputs) {
        if (output) {
            func_outputs.push_back(output_names.at(output_index), output.value());
        } else {
            throw std::invalid_argument(
                std::format("No value assigned to output {}", output_index + 1)
            );
        }
        ++output_index;
    }

    for (auto [instr, use_count] :
         zip(std::views::reverse(_instructions),
             std::views::reverse(_instruction_use_count))) {
        if (use_count > 0) {
            continue;
        }
        for (auto& input : instr.inputs) {
            int input_source = _local_sources.at(input.local_index);
            if (input_source != -1) {
                --_instruction_use_count.at(input_source);
            }
        }
    }

    std::vector<InstructionCall> filtered_instructions;
    for (auto [instr, use_count] : zip(_instructions, _instruction_use_count)) {
        if (use_count > 0) {
            filtered_instructions.push_back(instr);
        }
    }

    return Function(
        _inputs,
        func_outputs,
        _locals,
        {_globals.begin(), _globals.end()},
        filtered_instructions
    );
}

Value FunctionBuilder::input(int index) const {
    if (index < 0 || index >= _inputs.size()) {
        throw std::out_of_range(
            std::format(
                "Input index expected to be in range 0 to {}, got {}",
                _inputs.size() - 1,
                index
            )
        );
    }
    return _inputs.at(index);
}

ValueVec FunctionBuilder::input_range(int start_index, int end_index) const {
    if (start_index < 0 || start_index > _inputs.size()) {
        throw std::out_of_range(
            std::format(
                "Start index expected to be in range 0 to {}, got {}",
                _inputs.size(),
                start_index
            )
        );
    }
    if (end_index < start_index || end_index > _inputs.size()) {
        throw std::out_of_range(
            std::format(
                "End index expected to be in range {} to {}, got {}",
                start_index,
                _inputs.size() - 1,
                end_index
            )
        );
    }
    return ValueVec(_inputs.begin() + start_index, _inputs.begin() + end_index);
}

void FunctionBuilder::output(int index, Value value) {
    if (index < 0 || index >= _outputs.size()) {
        throw std::out_of_range(
            std::format(
                "Output index expected to be in range 0 to {}, got {}",
                _outputs.size() - 1,
                index
            )
        );
    }
    auto& out_type = _output_types.at(index);
    if (out_type.dtype != value.type.dtype || out_type.shape != value.type.shape) {
        throw std::invalid_argument(
            std::format("Wrong output type for output {}", index + 1)
        );
    }
    register_local(value);
    _outputs.at(index) = value;
    int value_source = _local_sources.at(value.local_index);
    if (value_source != -1) {
        ++_instruction_use_count.at(value_source);
    }
}

void FunctionBuilder::output_range(int start_index, const ValueVec& values) {
    if (start_index < 0 || start_index > _outputs.size() - values.size()) {
        throw std::out_of_range(
            std::format(
                "Start index expected to be in range 0 to {}, got {}",
                _outputs.size() - values.size(),
                start_index
            )
        );
    }
    int index = start_index;
    for (auto& value : values) {
        output(index, value);
        ++index;
    }
}

Value FunctionBuilder::global(
    const std::string& name, DataType dtype, const std::vector<int>& shape
) {
    Type type(dtype, BatchSize::one, shape);

    if (auto search = _globals.find(name); search != _globals.end()) {
        auto& found_global = search->second;
        if (type != found_global.type) {
            throw std::invalid_argument(
                std::format("Conflicting types for global {}", name)
            );
        }
        return found_global;
    }

    int local_index = _locals.size();
    Value new_global(type, local_index);
    _locals.push_back(new_global);
    _local_sources.push_back(-1);
    _globals[name] = new_global;
    return new_global;
}

Value FunctionBuilder::sum(const ValueVec& values) {
    if (values.size() == 0) {
        return 0.0;
    }
    auto result = values.at(0);
    for (auto value = values.begin() + 1; value != values.end(); ++value) {
        result = add(result, *value);
    }
    return result;
}

void FunctionBuilder::register_local(Value& val) {
    if (val.local_index != -1) {
        return;
    }

    auto find_literal = _literals.find(val.literal_value);
    if (find_literal == _literals.end()) {
        int local_index = _locals.size();
        val = Value(val.type, val.literal_value, local_index);
        _locals.push_back(val);
        _local_sources.push_back(-1);
        _literals[val.literal_value] = val;
    } else {
        val = find_literal->second;
    }
}

Value FunctionBuilder::product(const ValueVec& values) {
    switch (values.size()) {
    case 0:
        return 1.;
    case 1:
        return values.at(0);
    case 2:
        return mul(values.at(0), values.at(1));
    case 3:
        return mul(mul(values.at(0), values.at(1)), values.at(2));
    default:
        return reduce_product(stack(values));
    }
}

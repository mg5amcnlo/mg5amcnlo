#pragma once

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "instruction.hpp"
#include "madspace/util.hpp"

namespace madspace {

struct InstructionCall {
    InstructionPtr instruction;
    ValueVec inputs;
    ValueVec outputs;
    std::size_t stream_index;
};

class Function {
public:
    friend class FunctionBuilder;

    Function() = default;

    const NamedVector<Value>& inputs() const { return _inputs; }
    const NamedVector<Value>& outputs() const { return _outputs; }
    const ValueVec& locals() const { return _locals; }
    const std::vector<std::pair<std::string, Value>>& globals() const {
        return _globals;
    }
    const std::vector<InstructionCall>& instructions() const { return _instructions; }

    void save(const std::string& file) const;
    static Function load(const std::string& file);

private:
    Function(
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& outputs,
        const ValueVec& locals,
        const std::vector<std::pair<std::string, Value>>& globals,
        const std::vector<InstructionCall>& instructions
    ) :
        _inputs(inputs),
        _outputs(outputs),
        _locals(locals),
        _globals(globals),
        _instructions(instructions) {}

    NamedVector<Value> _inputs;
    NamedVector<Value> _outputs;
    ValueVec _locals;
    std::vector<std::pair<std::string, Value>> _globals;
    std::vector<InstructionCall> _instructions;

    friend Function sort_breadth_first(const Function& function);
};

std::ostream& operator<<(std::ostream& out, const Value& value);
std::ostream& operator<<(std::ostream& out, const ValueVec& list);
std::ostream& operator<<(std::ostream& out, const InstructionCall& call);
std::ostream& operator<<(std::ostream& out, const Function& func);

void to_json(nlohmann::json& j, const InstructionCall& call);
void to_json(nlohmann::json& j, const Function& call);
void from_json(const nlohmann::json& j, Function& call);

class FunctionBuilder {
public:
    FunctionBuilder(
        const NamedVector<Type>& _input_types, const NamedVector<Type>& _output_types
    );
    FunctionBuilder(const Function& function);
    Value input(int index) const;
    ValueVec input_range(int start_index, int end_index) const;
    void output(int index, Value value);
    void output_range(int start_index, const ValueVec& values);
    Value
    global(const std::string& name, DataType dtype, const std::vector<int>& shape);
    ValueVec instruction(const std::string& name, const ValueVec& args);
    ValueVec instruction(InstructionPtr instruction, const ValueVec& args);
    std::size_t current_stream() const { return _current_stream; }
    void set_current_stream(std::size_t stream_index) {
        _current_stream = stream_index;
    }
    Function function();

    Value sum(const ValueVec& values);
    Value product(const ValueVec& values);

#include "function_builder_mixin.inc"

private:
    NamedVector<Type> _output_types;
    NamedVector<Value> _inputs;
    std::vector<std::optional<Value>> _outputs;
    std::map<LiteralValue, Value> _literals;
    ValueVec _locals;
    std::unordered_map<std::string, Value> _globals;
    std::vector<InstructionCall> _instructions;
    std::map<std::vector<std::size_t>, std::vector<std::size_t>> _instruction_cache;
    std::vector<int> _local_sources;
    std::vector<std::size_t> _instruction_use_count;
    std::size_t _current_stream;

    void register_local(Value& val);
};

} // namespace madspace

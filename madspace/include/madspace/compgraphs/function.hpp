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

/**
 * One recorded instruction in a @ref Function. It pairs the operation to run
 * with the values it reads and writes.
 */
struct InstructionCall {
    /// The operation to perform.
    InstructionPtr instruction;
    /// Input values, in the instruction's argument order.
    ValueVec inputs;
    /// Output values it produces.
    ValueVec outputs;
    /// Execution stream this call belongs to (0 is the main stream).
    std::size_t stream_index;
};

/**
 * An immutable compute graph produced by @ref FunctionBuilder.
 *
 * A function has named `inputs` and `outputs`, a list of intermediate `locals`,
 * named `globals` for trainable or externally supplied tensors, and the ordered
 * list of @ref InstructionCall that computes the outputs from the inputs. Run
 * one with a `FunctionRuntime`, or `save` it to disk and `load` it back.
 */
class Function {
public:
    friend class FunctionBuilder;

    Function() = default;

    /// Named graph inputs.
    const NamedVector<Value>& inputs() const { return _inputs; }
    /// Named graph outputs.
    const NamedVector<Value>& outputs() const { return _outputs; }
    /// Intermediate values, one per instruction output.
    const ValueVec& locals() const { return _locals; }
    /// Named global tensors: trainable parameters and external inputs.
    const std::vector<std::pair<std::string, Value>>& globals() const {
        return _globals;
    }
    /// The instruction calls in execution order.
    const std::vector<InstructionCall>& instructions() const { return _instructions; }

    /// Serialize the function to @p file.
    void save(const std::string& file) const;
    /// Load a function previously written by `save`.
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
    friend Function fuse_instructions(const Function& function);
};

std::ostream& operator<<(std::ostream& out, const Value& value);
std::ostream& operator<<(std::ostream& out, const ValueVec& list);
std::ostream& operator<<(std::ostream& out, const InstructionCall& call);
std::ostream& operator<<(std::ostream& out, const Function& func);

void to_json(nlohmann::json& j, const InstructionCall& call);
void to_json(nlohmann::json& j, const Function& call);
void from_json(const nlohmann::json& j, Function& call);

/**
 * Records instructions and freezes them into a @ref Function.
 *
 * Construct it with the input and output types, read the graph inputs with
 * `input`, add nodes by calling the instruction methods (for example `add`,
 * `stack` or `two_body_decay_com`, listed below), wire the results to the
 * outputs with `output`, then call `function` to obtain the finished
 * @ref Function. The instruction methods are generated from
 * `madspace/instruction_set.yaml`.
 */
class FunctionBuilder {
public:
    /// Start an empty builder with the given input and output types.
    FunctionBuilder(
        const NamedVector<Type>& _input_types, const NamedVector<Type>& _output_types
    );
    /// Start a builder pre-populated with an existing function.
    FunctionBuilder(const Function& function);
    /// The graph input at position @p index.
    Value input(int index) const;
    /// The graph inputs in the half-open range [@p start_index, @p end_index).
    ValueVec input_range(int start_index, int end_index) const;
    /// Wire @p value to the graph output at position @p index.
    void output(int index, Value value);
    /// Wire @p values to the graph outputs starting at @p start_index.
    void output_range(int start_index, const ValueVec& values);
    /// Declare a named global tensor and return a value reading it.
    /// @param name global name
    /// @param dtype element data type
    /// @param shape static shape, without the batch dimension
    Value
    global(const std::string& name, DataType dtype, const std::vector<int>& shape);
    /// Append the instruction @p name with arguments @p args; returns its outputs.
    ValueVec instruction(const std::string& name, const ValueVec& args);
    /// Append @p instruction with arguments @p args; returns its outputs.
    ValueVec instruction(InstructionPtr instruction, const ValueVec& args);
    /// Calls the float32 or float64 variant `<name>_float32` / `<name>_float64` of
    /// an instruction, chosen by the first non-literal float argument. Float64
    /// literals are converted for a float32 call.
    ValueVec precision_instruction(const std::string& name, const ValueVec& args);
    /// Index of the execution stream new instructions are added to.
    std::size_t current_stream() const { return _current_stream; }
    /// Select the execution stream new instructions are added to.
    void set_current_stream(std::size_t stream_index) {
        _current_stream = stream_index;
    }
    /// Freeze the recorded graph into an immutable @ref Function.
    Function function();

    /// Sum of @p values, element by element.
    Value sum(const ValueVec& values);
    /// Product of @p values, element by element.
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

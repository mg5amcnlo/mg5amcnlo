import os

import yaml


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("instruction_set.yaml") as f:
        data = list(yaml.safe_load_all(f))

    commands = {}
    for sec in data:
        commands.update({key: value for key, value in sec.items() if key != "title"})
    for i, cmd in enumerate(commands.values()):
        cmd["opcode"] = i
    sections = [
        (sec["title"], [key for key in sec.keys() if key != "title"])
        for sec in data[1:]
    ]

    function_builder_mixin(commands)
    instruction_set_python(commands)
    instruction_set_mixin(commands)
    runtime_mixin(commands, "cpu")
    runtime_backward_mixin(commands, "cpu")
    runtime_mixin(commands, "gpu")
    runtime_backward_mixin(commands, "gpu")


def write_autogen(f):
    f.write(
        "// This file was automatically generated based on instruction_set.yaml\n"
        "// Do not modify its content directly\n\n"
    )


def function_builder_mixin(commands):
    with open("include/madspace/compgraphs/function_builder_mixin.inc", "w") as f:
        write_autogen(f)
        first = True
        for name, cmd in commands.items():
            if first:
                first = False
            else:
                f.write("\n")
            if cmd["inputs"] == "any":
                parameters = "ValueVec args"
                instruction_call = f'instruction("{name}", args)'
            else:
                parameters = ", ".join(f"Value {arg['name']}" for arg in cmd["inputs"])
                arguments = ", ".join(arg["name"] for arg in cmd["inputs"])
                instruction_call = f'instruction("{name}", {{{arguments}}})'

            if cmd["outputs"] == "any":
                return_type = "ValueVec"
                func_body = f"    return {instruction_call};"
            else:
                n_outputs = len(cmd["outputs"])
                if n_outputs == 0:
                    return_type = "void"
                    func_body = f"    {instruction_call};"
                elif n_outputs == 1:
                    return_type = "Value"
                    func_body = f"    return {instruction_call}[0];"
                else:
                    return_type = f"std::array<Value, {n_outputs}>"
                    return_array = ", ".join(
                        f"output_vector[{i}]" for i in range(n_outputs)
                    )
                    func_body = (
                        f"    auto output_vector = {instruction_call};\n"
                        f"    return {{{return_array}}};"
                    )

            f.write(f"{return_type} {name}({parameters}) {{\n{func_body}\n}}\n")


def instruction_set_python(commands):
    with open("src/python/instruction_set.hpp", "w") as f:
        write_autogen(f)
        f.write(
            "#pragma once\n\n"
            "#include <pybind11/pybind11.h>\n"
            "#include <pybind11/stl.h>\n"
            '#include "madspace/compgraphs.hpp"\n\n'
            "namespace py = pybind11;\n"
            "using madspace::FunctionBuilder;\n\n"
            "namespace {\n\n"
            "void add_instructions(py::classh<FunctionBuilder>& fb) {\n"
        )

        for name, cmd in commands.items():
            f.write(f'    fb.def("{name}", &FunctionBuilder::{name}')
            if cmd["inputs"] == "any":
                f.write(', py::arg("args")')
            else:
                for arg in cmd["inputs"]:
                    f.write(f', py::arg("{arg["name"]}")')
            f.write(");\n")

        f.write("}\n}\n")


def format_type(data):
    type = data["type"]
    if type[0] == "size":
        return f'{{DataType::dt_int, true, {{"{type[1]}"}}, true}}'

    dtype = f"DataType::dt_{type[0]}"
    single = len(type) > 1 and type[1] == "single"
    single_str = "true" if single else "false"
    shape = ", ".join(
        (
            str(item)
            if isinstance(item, int)
            else ("std::monostate{}" if item == "..." else f'"{item}"')
        )
        for item in type[single + 1 :]
    )
    return f"{{{dtype}, {single_str}, {{{shape}}}, false}}"


def instruction_set_mixin(commands):
    with (
        open("src/compgraphs/instruction_set_mixin.inc", "w") as f,
        open("include/madspace/compgraphs/opcode_mixin.inc", "w") as f_op,
    ):
        write_autogen(f)
        f.write("using SigType = SimpleInstruction::SigType;\n")

        f.write(
            "const auto mi = [](\n"
            "    std::string name,\n"
            "    int opcode,\n"
            "    bool differentiable,\n"
            "    std::initializer_list<SigType> inputs,\n"
            "    std::initializer_list<SigType> outputs\n"
            ") {\n"
            "    return InstructionOwner(new SimpleInstruction(\n"
            "        name, opcode, differentiable, inputs, outputs\n"
            "    ));\n"
            "};\n"
            "\n"
            "InstructionOwner instructions[] {\n"
        )
        first = True
        for name, cmd in commands.items():
            opcode = cmd["opcode"]
            differentiable = "true" if cmd.get("differentiable", False) else "true"
            if "class" in cmd:
                f.write(
                    f"    InstructionOwner(new {cmd['class']}("
                    f"{opcode}, {differentiable})),\n"
                )
            else:
                input_types = ", ".join(format_type(arg) for arg in cmd["inputs"])
                output_types = ", ".join(format_type(ret) for ret in cmd["outputs"])
                f.write(
                    f'    mi("{name}", {opcode}, {differentiable}, '
                    f"{{{input_types}}}, {{{output_types}}}),\n"
                )

            if first:
                first = False
            else:
                f_op.write(",\n")
            f_op.write(f"{name} = {opcode}")
        f.write("};\n")
        f_op.write("\n")


def runtime_mixin(commands, device):
    with open(f"src/{device}/runtime_mixin.inc", "w") as f:
        write_autogen(f)

        for name, cmd in commands.items():
            opcode = cmd["opcode"]
            if cmd.get("custom_op", False):
                func = f"op_{name}"
            else:
                n_inputs = len(cmd["inputs"])
                n_outputs = len(cmd["outputs"])
                dims = cmd.get("dims", 1)
                vectorized = cmd.get("vectorized", True)

                if device == "cpu":
                    if vectorized:
                        kernel = f"kernel_{name}<CpuTypes>, kernel_{name}<SimdTypes>"
                    else:
                        kernel = f"kernel_{name}<CpuTypes>, kernel_{name}<CpuTypes>"
                    device_arg = ", DeviceType"
                elif device == "gpu":
                    kernel = f"kernel_{name}<GpuTypes>"
                    device_arg = ""
                foreach_func = (
                    f"tensor_foreach_dynamic<{kernel}, {n_inputs}, {n_outputs}{device_arg}>"
                    if dims == 0
                    else f"tensor_foreach<{kernel}, {n_inputs}, {n_outputs}, {dims}{device_arg}>"
                )
                func = f"batch_foreach<{foreach_func}, {n_inputs}, {n_outputs}>"
            f.write(
                f"case {opcode}:\n"
                f"    {func}(instr, locals, device);\n"
                f"    break;\n"
            )


def runtime_backward_mixin(commands, device):
    with open(f"src/{device}/runtime_backward_mixin.inc", "w") as f:
        write_autogen(f)

        for name, cmd in commands.items():
            if not cmd.get("differentiable", False):
                continue
            opcode = cmd["opcode"]
            if cmd.get("custom_op", False):
                f.write(
                    f"case {opcode}:\n"
                    f"    backward_op_{name}(instr, locals, local_grads, device);\n"
                    f"    break;\n"
                )
            else:
                n_inputs = len(cmd["inputs"])
                n_outputs = len(cmd["outputs"])
                in_stored = [
                    i
                    for i, arg in enumerate(cmd["inputs"])
                    if arg.get("backward_arg", False)
                ]
                out_stored = [
                    i
                    for i, arg in enumerate(cmd["outputs"])
                    if arg.get("backward_arg", False)
                ]
                if len(in_stored) + len(out_stored) == 0:
                    in_stored = list(range(n_inputs))
                in_stored_str = ",".join(str(i) for i in in_stored)
                out_stored_str = ",".join(str(i) for i in out_stored)
                vectorized = cmd.get("vectorized", True)

                if device == "cpu":
                    if vectorized:
                        kernel = (
                            f"backward_kernel_{name}<CpuTypes>, "
                            f"backward_kernel_{name}<SimdTypes>"
                        )
                    else:
                        kernel = (
                            f"backward_kernel_{name}<CpuTypes>, "
                            f"backward_kernel_{name}<CpuTypes>"
                        )
                    device_arg = ", DeviceType"
                elif device == "gpu":
                    kernel = f"backward_kernel_{name}<GpuTypes>"
                    device_arg = ""

                dims = cmd.get("dims", 1)
                n_args = len(in_stored) + len(out_stored) + n_outputs
                foreach_func = (
                    f"tensor_foreach_dynamic<{kernel}, {n_args}, {n_inputs}{device_arg}>"
                    if dims == 0
                    else f"tensor_foreach<{kernel}, {n_args}, {n_inputs}, {dims}{device_arg}>"
                )
                func = (
                    f"backward_batch_foreach<{foreach_func}, {n_inputs}, {n_outputs}, "
                    f"{len(in_stored)}, {len(out_stored)}>"
                )
                f.write(
                    f"case {opcode}:\n"
                    f"    {func}(instr, locals, local_grads, "
                    f"{{{in_stored_str}}}, {{{out_stored_str}}}, device);\n"
                    f"    break;\n"
                )


if __name__ == "__main__":
    main()

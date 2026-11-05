import argparse
import os
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Root directory to write the generated headers into.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()

    os.chdir(SCRIPT_DIR)
    with open("instruction_set.yaml") as f:
        data = list(yaml.safe_load_all(f))

    commands = {}
    for sec in data:
        commands.update({key: value for key, value in sec.items() if key != "title"})
    commands = expand_precisions(commands)
    for i, cmd in enumerate(commands.values()):
        cmd["opcode"] = i
    sections = [
        (sec["title"], [key for key in sec.keys() if key != "title"])
        for sec in data[1:]
    ]

    for sub_dir in ("compgraphs", "python", "cpu", "gpu"):
        (out_dir / sub_dir).mkdir(parents=True, exist_ok=True)

    function_builder_mixin(commands, out_dir)
    instruction_set_python(commands, out_dir)
    instruction_set_mixin(commands, out_dir)
    runtime_mixin(commands, "cpu", out_dir)
    runtime_backward_mixin(commands, "cpu", out_dir)
    runtime_mixin(commands, "gpu", out_dir)
    runtime_backward_mixin(commands, "gpu", out_dir)
    kernel_checks(commands, out_dir)


# C++ element type and kernel policy suffix for each float precision
PRECISIONS = {"float32": ("float", "32"), "float64": ("double", "")}


def expand_precisions(commands):
    """Split each instruction with a `precision` list into one instruction per
    precision, named `<name>_<precision>`. In these, `float` in a type means the
    instruction's precision. The expanded entries keep the original name in
    `base_name` and the kernel policy suffix in `types_suffix`."""
    expanded = {}
    for name, cmd in commands.items():
        precisions = cmd.get("precision")
        if precisions is None:
            cmd["base_name"] = name
            cmd["types_suffix"] = ""
            cmd["dispatch"] = None
            expanded[name] = cmd
            continue
        for precision in precisions:
            ctype, suffix = PRECISIONS[precision]
            sub_cmd = dict(cmd)
            for key in ("inputs", "outputs"):
                if sub_cmd[key] == "any":
                    raise ValueError(f"{name}: precision needs explicit {key}")
                sub_cmd[key] = [
                    {
                        **arg,
                        "type": (
                            [precision, *arg["type"][1:]]
                            if arg.get("type", [None])[0] == "float"
                            else arg.get("type")
                        ),
                    }
                    for arg in cmd[key]
                ]
            sub_cmd["base_name"] = name
            sub_cmd["types_suffix"] = suffix
            sub_cmd["ctype"] = ctype
            sub_cmd["doc_cmd"] = cmd
            # the builder method is generated once, for the first precision
            sub_cmd["dispatch"] = precisions if precision == precisions[0] else []
            expanded[f"{name}_{precision}"] = sub_cmd
    return expanded


def write_autogen(f):
    f.write(
        "// This file was automatically generated based on instruction_set.yaml\n"
        "// Do not modify its content directly\n\n"
    )


def format_shape(entry):
    """Human-readable shape for one instruction input/output, from its `type`.

    `[float]` -> "float, shape (batch,)"; `[float, 4]` -> "(batch, 4)";
    `[int, single]` -> "int, scalar"; `[size, name]` -> "batch-size list".
    An entry without a `type` key yields None.
    """
    type = entry.get("type")
    if not type:
        return None
    if type[0] == "size":
        return "batch-size list"
    dtype = type[0]
    single = len(type) > 1 and type[1] == "single"
    dims = [str(d) for d in type[1 + single :]]
    if single:
        shape = "scalar" if not dims else f"shape ({', '.join(dims)})"
    else:
        inner = ", ".join(["batch", *dims])
        shape = f"shape ({inner},)" if not dims else f"shape ({inner})"
    return f"{dtype}, {shape}"


def instruction_brief(name, cmd):
    """The `desc` from the YAML, or a templated fallback for an empty one."""
    desc = (cmd.get("desc") or "").strip()
    if desc:
        desc = " ".join(desc.split())
        return desc if desc.endswith(".") else desc + "."
    return f"The `{name}` instruction."


def _entry_text(entry, fallback):
    """`(shape) description` for one input/output, never empty."""
    shape = format_shape(entry)
    desc = " ".join((entry.get("desc") or "").split())
    parts = ([f"({shape})"] if shape else []) + ([desc] if desc else [])
    return " ".join(parts) if parts else fallback


def doc_comment(name, cmd):
    """A Doxygen `/** */` block for one generated FunctionBuilder method."""
    lines = [f" * {instruction_brief(name, cmd)}"]
    inputs = cmd["inputs"]
    if inputs == "any":
        lines.append(" * @param args variadic input values.")
    else:
        for arg in inputs:
            lines.append(f" * @param {arg['name']} {_entry_text(arg, 'input value.')}")
    outputs = cmd["outputs"]
    if outputs == "any":
        lines.append(" * @return the list of output values.")
    elif len(outputs) == 1:
        lines.append(f" * @return {_entry_text(outputs[0], 'the output value.')}")
    elif len(outputs) > 1:
        lines.append(" * Returns, in order:")
        for out in outputs:
            lines.append(f" * - `{out['name']}` - {_entry_text(out, 'output value.')}")
    return "/**\n" + "\n".join(lines) + "\n */\n"


def function_builder_mixin(commands, out_dir):
    with open(out_dir / "compgraphs" / "function_builder_mixin.inc", "w") as f:
        write_autogen(f)
        first = True
        for name, cmd in commands.items():
            if cmd["dispatch"] == []:
                continue
            if first:
                first = False
            else:
                f.write("\n")
            base_name = cmd["base_name"]
            f.write(doc_comment(base_name, cmd.get("doc_cmd", cmd)))
            if cmd["inputs"] == "any":
                parameters = "ValueVec args"
                instruction_call = f'instruction("{name}", args)'
            elif cmd["dispatch"]:
                parameters = ", ".join(f"Value {arg['name']}" for arg in cmd["inputs"])
                arguments = ", ".join(arg["name"] for arg in cmd["inputs"])
                instruction_call = (
                    f'precision_instruction("{base_name}", {{{arguments}}})'
                )
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

            f.write(f"{return_type} {base_name}({parameters}) {{\n{func_body}\n}}\n")


def instruction_set_python(commands, out_dir):
    with open(out_dir / "python" / "instruction_set.hpp", "w") as f:
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
            if cmd["dispatch"] == []:
                continue
            name = cmd["base_name"]
            f.write(f'    fb.def("{name}", &FunctionBuilder::{name}')
            if cmd["inputs"] == "any":
                f.write(', py::arg("args")')
            else:
                for arg in cmd["inputs"]:
                    f.write(f', py::arg("{arg["name"]}")')
            f.write(");\n")

        f.write("}\n}\n")


# yaml shorthands for the default element types
DTYPE_NAMES = {"float": "float64", "int": "int32"}


def format_type(data):
    type = data["type"]
    if type[0] == "size":
        return f'{{DataType::dt_int32, true, {{"{type[1]}"}}, true}}'

    dtype = f"DataType::dt_{DTYPE_NAMES.get(type[0], type[0])}"
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


def instruction_set_mixin(commands, out_dir):
    with (
        open(out_dir / "compgraphs" / "instruction_set_mixin.inc", "w") as f,
        open(out_dir / "compgraphs" / "opcode_mixin.inc", "w") as f_op,
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
            differentiable = "true" if cmd.get("differentiable", False) else "false"
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


def custom_op_func(prefix, cmd):
    """The C++ function of a custom op. custom_op can name an implementation shared
    by several instructions instead of True, with the leading template arguments
    custom_op_args."""
    custom_op = cmd["custom_op"]
    name = custom_op if isinstance(custom_op, str) else cmd["base_name"]
    template_args = list(cmd.get("custom_op_args", []))
    if "ctype" in cmd:
        template_args.append(cmd["ctype"])
    func = f"{prefix}{name}"
    return f"{func}<{', '.join(template_args)}>" if template_args else func


def runtime_mixin(commands, device, out_dir):
    with open(out_dir / device / "runtime_mixin.inc", "w") as f:
        write_autogen(f)

        for name, cmd in commands.items():
            opcode = cmd["opcode"]
            base_name = cmd["base_name"]
            suffix = cmd["types_suffix"]
            if cmd.get("custom_op", False):
                func = custom_op_func("op_", cmd)
            else:
                n_inputs = len(cmd["inputs"])
                n_outputs = len(cmd["outputs"])
                dims = cmd.get("dims", 1)
                vectorized = cmd.get("vectorized", True)

                kernel_name = f"kernel_{base_name}"
                if device == "cpu":
                    simd = "Simd" if vectorized else "Cpu"
                    kernel = (
                        f"{kernel_name}<CpuTypes{suffix}>, "
                        f"{kernel_name}<{simd}Types{suffix}>"
                    )
                    device_arg = ", DeviceType"
                elif device == "gpu":
                    kernel = f"{kernel_name}<GpuTypes{suffix}>"
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


def runtime_backward_mixin(commands, device, out_dir):
    with open(out_dir / device / "runtime_backward_mixin.inc", "w") as f:
        write_autogen(f)

        for name, cmd in commands.items():
            if not cmd.get("differentiable", False):
                continue
            opcode = cmd["opcode"]
            base_name = cmd["base_name"]
            suffix = cmd["types_suffix"]
            if cmd.get("custom_op", False):
                func = custom_op_func("backward_op_", cmd)
                f.write(
                    f"case {opcode}:\n"
                    f"    {func}(instr, locals, local_grads, device);\n"
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

                kernel_name = f"backward_kernel_{base_name}"
                if device == "cpu":
                    simd = "Simd" if vectorized else "Cpu"
                    kernel = (
                        f"{kernel_name}<CpuTypes{suffix}>, "
                        f"{kernel_name}<{simd}Types{suffix}>"
                    )
                    device_arg = ", DeviceType"
                elif device == "gpu":
                    kernel = f"{kernel_name}<GpuTypes{suffix}>"
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


def is_single_input(arg):
    return (
        isinstance(arg.get("type"), list)
        and len(arg["type"]) > 1
        and (arg["type"][1] == "single")
    )


def kernel_checks(commands, out_dir):
    """Compile-time checks that the parameters of the vectorized kernels use the
    single types (FSIn, ISIn) exactly for the inputs marked as single in the yaml.
    A mismatch would make the CPU runtime fall back to the scalar kernels."""
    with open(out_dir / "cpu" / "kernel_checks.inc", "w") as f:
        write_autogen(f)

        def check(kernel, index, single, what):
            f.write(
                f"static_assert(\n"
                f"    single_params<decltype(&{kernel}<SimdTypes{suffix}>)>"
                f"::value[{index}] == "
                f"{'true' if single else 'false'},\n"
                f'    "{kernel}: parameter {index} ({what}) must '
                f'{"" if single else "not "}be single"\n'
                f");\n"
            )

        for name, cmd in commands.items():
            if cmd.get("custom_op", False) or not cmd.get("vectorized", True):
                continue
            suffix = cmd["types_suffix"]
            name = cmd["base_name"]
            inputs = cmd["inputs"]
            if is_single_input(inputs[0]):
                raise ValueError(
                    f"{name}: the first input of a kernel can not be single"
                )
            for i, arg in enumerate(inputs):
                check(f"kernel_{name}", i, is_single_input(arg), arg["name"])

            if not cmd.get("differentiable", False):
                continue
            in_stored = [
                i for i, arg in enumerate(inputs) if arg.get("backward_arg", False)
            ]
            out_stored = [
                i
                for i, arg in enumerate(cmd["outputs"])
                if arg.get("backward_arg", False)
            ]
            if len(in_stored) + len(out_stored) == 0:
                in_stored = list(range(len(inputs)))
            for k, i in enumerate(in_stored):
                check(
                    f"backward_kernel_{name}",
                    k,
                    is_single_input(inputs[i]),
                    inputs[i]["name"],
                )


if __name__ == "__main__":
    main()

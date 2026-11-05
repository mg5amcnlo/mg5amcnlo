#!/usr/bin/env rust-script
// SPDX-License-Identifier: 0BSD
//! ```cargo
//! [package]
//! edition = "2021"
//! ```

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
mod rusticol {
    include!(env!("RUSTICOL_RUST_SOURCE"));
}

use rusticol::{ParameterUpdate, PhysicsMetadata, ResolvedEvaluation, Runtime, Selectors};

#[derive(Default)]
struct Options {
    process: Option<String>,
    model_parameters: Option<String>,
    kinematics: Option<String>,
    overrides: Vec<ParameterUpdate>,
    precision: u32,
    json: bool,
    help: bool,
}

fn sdk<T>(result: rusticol::Result<T>) -> Result<T, String> {
    result.map_err(|error| error.to_string())
}

fn usage() -> &'static str {
    "usage: check_standalone [--process ID|EXPRESSION] [--model-parameters PATH] \
[--kinematics PATH] [--set-parameter NAME REAL IMAG] [--precision 16] [--json]"
}

fn parse_options() -> Result<Options, String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let mut options = Options {
        precision: 16,
        ..Options::default()
    };
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        match argument.as_str() {
            "--" => {
                index += 1;
            }
            "--process" => {
                options.process = Some(required_argument(&arguments, index, 1)?.to_owned());
                index += 2;
            }
            "--model-parameters" => {
                options.model_parameters =
                    Some(required_argument(&arguments, index, 1)?.to_owned());
                index += 2;
            }
            "--kinematics" => {
                options.kinematics =
                    Some(required_argument(&arguments, index, 1)?.to_owned());
                index += 2;
            }
            "--set-parameter" => {
                let name = required_argument(&arguments, index, 1)?.to_owned();
                let real = parse_f64(
                    required_argument(&arguments, index, 2)?,
                    "real model-parameter component",
                )?;
                let imaginary = parse_f64(
                    required_argument(&arguments, index, 3)?,
                    "imaginary model-parameter component",
                )?;
                options
                    .overrides
                    .push(ParameterUpdate::new(name, real, imaginary));
                index += 4;
            }
            "--precision" => {
                options.precision = required_argument(&arguments, index, 1)?
                    .parse()
                    .map_err(|_| "invalid --precision value".to_owned())?;
                index += 2;
            }
            "--json" => {
                options.json = true;
                index += 1;
            }
            "--help" | "-h" => {
                options.help = true;
                index += 1;
            }
            _ => return Err(format!("unknown option: {argument}")),
        }
    }
    if options.precision != 16 {
        return Err(
            "the Rust Rusticol API supports only double precision (--precision 16)".to_owned(),
        );
    }
    Ok(options)
}

fn required_argument<'a>(
    arguments: &'a [String],
    option_index: usize,
    offset: usize,
) -> Result<&'a str, String> {
    arguments
        .get(option_index + offset)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value after {}", arguments[option_index]))
}

fn parse_f64(value: &str, description: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid {description}"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(format!("invalid {description}"))
    }
}

fn artifact_root() -> Result<PathBuf, String> {
    let current =
        env::current_dir().map_err(|error| format!("cannot read current directory: {error}"))?;
    if current.join("artifact.json").is_file() {
        return Ok(current);
    }

    let executable = env::current_exe()
        .map_err(|error| format!("cannot locate check_standalone executable: {error}"))?;
    for ancestor in executable.ancestors().skip(1) {
        if ancestor.join("artifact.json").is_file() {
            return Ok(ancestor.to_path_buf());
        }
    }

    let language_dir = executable.parent();
    let build_artifact = language_dir.and_then(Path::parent);
    let build_root = build_artifact.and_then(Path::parent);
    if build_root.and_then(Path::file_name) == Some(".pyamplicol-api-build".as_ref()) {
        if let (Some(parent), Some(name)) = (
            build_root.and_then(Path::parent),
            build_artifact.and_then(Path::file_name),
        ) {
            let candidate = parent.join(name);
            if candidate.join("artifact.json").is_file() {
                return Ok(candidate);
            }
        }
    }

    Err("run check_standalone from a generated artifact directory".to_owned())
}

fn load_validation_point(
    path: &Path,
    process_key: &str,
    external_count: usize,
) -> Result<Option<Vec<f64>>, String> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
    };
    let mut lines = contents.lines();
    if lines.next() != Some("RUSTICOL_VALIDATION_POINTS_V1") {
        return Err("unsupported validation_points.dat format".to_owned());
    }
    for line in lines {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() < 2 || fields[0] != process_key {
            continue;
        }
        let row_count = fields[1]
            .parse::<usize>()
            .map_err(|_| "invalid validation point row".to_owned())?;
        if row_count != external_count || fields.len() != 2 + 4 * external_count {
            return Err("validation point has an incompatible external-particle count".to_owned());
        }
        let momenta = fields[2..]
            .iter()
            .map(|value| parse_f64(value, "validation-point component"))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Some(momenta));
    }
    Ok(None)
}

fn reorder_validation_point(
    representative: Vec<f64>,
    permutation: &[usize],
) -> Result<Vec<f64>, String> {
    if representative.len() != 4 * permutation.len() {
        return Err("validation point does not match external permutation".to_owned());
    }
    let mut public_point = vec![0.0; representative.len()];
    let mut seen = vec![false; permutation.len()];
    for (representative_index, &public_index) in permutation.iter().enumerate() {
        if public_index >= permutation.len() || seen[public_index] {
            return Err("runtime returned an invalid external permutation".to_owned());
        }
        seen[public_index] = true;
        public_point[4 * public_index..4 * public_index + 4].copy_from_slice(
            &representative[4 * representative_index..4 * representative_index + 4],
        );
    }
    Ok(public_point)
}

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1f}' => {
                write!(&mut output, "\\u{:04x}", character as u32)
                    .expect("writing to a String cannot fail");
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn write_i32_array(output: &mut String, values: &[i32]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to a String cannot fail");
    }
    output.push(']');
}

fn write_usize_array(output: &mut String, values: &[usize]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to a String cannot fail");
    }
    output.push(']');
}

fn write_f64_array(output: &mut String, values: &[f64]) -> Result<(), String> {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err("matrix-element output is not finite".to_owned());
        }
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to a String cannot fail");
    }
    output.push(']');
    Ok(())
}

fn write_common_json(output: &mut String, metadata: &PhysicsMetadata) {
    write!(
        output,
        "\"process\":{},\"process_key\":{},\"color_accuracy\":{}",
        json_string(&metadata.process),
        json_string(&metadata.process_key),
        json_string(&metadata.color_accuracy),
    )
    .expect("writing to a String cannot fail");

    output.push_str(",\"external_particles\":[");
    for (index, particle) in metadata.external_particles.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"index\":{},\"pdg\":{}}}",
            particle.index, particle.pdg
        )
        .expect("writing to a String cannot fail");
    }

    output.push_str("],\"helicities\":[");
    for (index, helicity) in metadata.helicities.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"id\":{},\"helicities\":",
            json_string(&helicity.id)
        )
        .expect("writing to a String cannot fail");
        write_i32_array(output, &helicity.helicities);
        output.push('}');
    }

    output.push_str("],\"colors\":[");
    for (index, color) in metadata.colors.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"id\":{},\"kind\":{},\"word\":",
            json_string(&color.id),
            json_string(&color.kind),
        )
        .expect("writing to a String cannot fail");
        write_usize_array(output, &color.word);
        output.push('}');
    }
    output.push(']');
}

fn available_json(
    metadata: &PhysicsMetadata,
    resolved: &ResolvedEvaluation,
    explicit_total: f64,
    total: f64,
) -> Result<String, String> {
    let (_, helicity_count, color_count) = resolved.shape();
    let mut output = "{\"language\":\"rust\",\"available\":true,\"precision\":16,".to_owned();
    write_common_json(&mut output, metadata);
    write!(
        &mut output,
        ",\"shape\":[1,{helicity_count},{color_count}],\"values\":"
    )
    .expect("writing to a String cannot fail");
    write_f64_array(&mut output, resolved.values())?;
    output.push_str(",\"resolved_sum\":");
    write_f64_array(&mut output, &[explicit_total])?;
    output.push_str(",\"compatibility_total\":");
    write_f64_array(&mut output, &[total])?;
    output.push_str("}\n");
    Ok(output)
}

fn unavailable_json(metadata: &PhysicsMetadata) -> String {
    let mut output = "{\"language\":\"rust\",\"available\":false,".to_owned();
    write_common_json(&mut output, metadata);
    output.push_str(",\"diagnostic\":\"no bundled validation point is available\"}\n");
    output
}

fn run() -> Result<(), String> {
    let options = parse_options()?;
    if options.help {
        println!("{}", usage());
        return Ok(());
    }
    let root = artifact_root()?;
    let model_parameters = options.model_parameters.as_deref().map(Path::new);
    let mut runtime = sdk(Runtime::load(
        &root,
        options.process.as_deref(),
        model_parameters,
    ))?;
    sdk(runtime.set_model_parameters(&options.overrides))?;
    let metadata = sdk(runtime.physics())?;
    let point = if let Some(path) = options.kinematics.as_deref() {
        Some(sdk(runtime.load_kinematics_json(path))?)
    } else {
        load_validation_point(
            &root.join("API/validation_points.dat"),
            &sdk(runtime.representative_process_key())?,
            metadata.external_particles.len(),
        )?
        .map(|point| {
            let permutation = sdk(runtime.external_permutation())?;
            reorder_validation_point(point, &permutation)
        })
        .transpose()?
    };
    let Some(momenta) = point else {
        if options.json {
            print!("{}", unavailable_json(&metadata));
        } else {
            println!("process: {}", metadata.process);
            println!("no bundled validation point is available; metadata load succeeded");
        }
        return Ok(());
    };

    let totals = sdk(runtime.evaluate_f64(&momenta, 1))?;
    let total = *totals
        .first()
        .ok_or_else(|| "Rusticol returned no compatibility total".to_owned())?;
    let resolved = sdk(runtime.evaluate_resolved_f64(&momenta, 1, &Selectors::all()))?;
    let (_, helicity_count, color_count) = resolved.shape();
    if helicity_count != metadata.helicities.len() || color_count != metadata.colors.len() {
        return Err("resolved shape does not match runtime metadata".to_owned());
    }
    let resolved_totals = resolved.totals();
    let explicit_total = *resolved_totals
        .first()
        .ok_or_else(|| "Rusticol returned no resolved total".to_owned())?;
    if !explicit_total.is_finite() || !total.is_finite() {
        return Err("matrix-element output is not finite".to_owned());
    }
    let scale = total.abs().max(1.0);
    if (explicit_total - total).abs() > 1.0e-12 * scale {
        return Err("resolved components do not reproduce the compatibility total".to_owned());
    }

    if options.json {
        print!(
            "{}",
            available_json(&metadata, &resolved, explicit_total, total)?
        );
        return Ok(());
    }

    println!("process: {} [{}]", metadata.process, metadata.process_key);
    println!("resolved shape: (1, {helicity_count}, {color_count})");
    for (helicity_index, helicity) in resolved.helicities().iter().enumerate() {
        for (color_index, color) in resolved.colors().iter().enumerate() {
            let value = resolved
                .get(0, helicity_index, color_index)
                .ok_or_else(|| "resolved result index is out of range".to_owned())?;
            println!("  {}  {}  {}", helicity.id, color.id, value);
        }
    }
    println!("explicit resolved sum: {explicit_total}");
    println!("compatibility total:   {total}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("check_standalone: {error}");
        std::process::exit(1);
    }
}

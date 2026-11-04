// SPDX-License-Identifier: 0BSD

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn copy_tree(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = destination.join(entry.file_name());
        if file_type.is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SDK staging may not contain symlinks",
            ));
        }
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), target)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SDK staging must contain regular files only",
            ));
        }
    }
    Ok(())
}

fn main() {
    pyo3_build_config::add_extension_module_link_args();
    println!("cargo:rerun-if-env-changed=PYAMPLICOL_SDK_STAGING");
    println!("cargo:rerun-if-env-changed=PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let sdk_output = output.join("_sdk");
    match fs::remove_dir_all(&sdk_output) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => panic!("remove previous Rusticol SDK from OUT_DIR: {error}"),
    }
    let Some(staging) = env::var_os("PYAMPLICOL_SDK_STAGING") else {
        return;
    };
    let source = PathBuf::from(staging);
    copy_tree(&source, &sdk_output).expect("copy validated Rusticol SDK into OUT_DIR");
}

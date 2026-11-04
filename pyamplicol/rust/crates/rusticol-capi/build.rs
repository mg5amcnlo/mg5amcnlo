// SPDX-License-Identifier: 0BSD

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let profile = env::var("PROFILE").expect("PROFILE");
    let target = env::var("TARGET").expect("TARGET");
    let payload = format!(
        "{{\n  \"schema_version\": 1,\n  \"target\": \"{target}\",\n  \"profile\": \"{profile}\",\n  \"native_static_libs\": []\n}}\n"
    );
    fs::write(output.join("rusticol-link.json"), payload).expect("write link metadata");
}

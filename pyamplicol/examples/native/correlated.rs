// SPDX-License-Identifier: 0BSD
// Set RUSTICOL_RUST_SOURCE to `rusticol-config --json`'s rust_source when compiling.
#[allow(dead_code)]
mod rusticol {
    include!(env!("RUSTICOL_RUST_SOURCE"));
}

use rusticol::{Complex64, CorrelatedRequest, Runtime, SpinCorrelationVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: correlated_rust ARTIFACT PROCESS".into());
    }
    let mut runtime = Runtime::load(&args[1], Some(&args[2]), None)?;
    let ids = runtime.color_correlation_ids()?;
    for id in ["born", "cascade-interference"] {
        if !ids.iter().any(|item| item == id) {
            return Err("required colour correlation is not registered".into());
        }
    }
    assert!(!runtime.color_correlation_catalogue_json()?.is_empty());
    let momenta = [
        400., 0., 0., 400., 400., 0., 0., -400., 300., 300., 0., 0., 250., -150., 200., 0., 250.,
        -150., -200., 0., 400., 0., 0., 400., 400., 0., 0., -400., 300., 0., 300., 0., 250., 200.,
        -150., 0., 250., -200., -150., 0.,
    ];
    let spins = vec![SpinCorrelationVector {
        leg: 5,
        components: vec![[
            Complex64::default(),
            Complex64::default(),
            Complex64::default(),
            Complex64::new(1., 0.),
        ]],
    }];
    let requests = [
        CorrelatedRequest {
            color_correlation: "born".into(),
            spin_vectors: Some(vec![]),
        },
        CorrelatedRequest {
            color_correlation: "cascade-interference".into(),
            spin_vectors: Some(vec![]),
        },
        CorrelatedRequest {
            color_correlation: "cascade-interference".into(),
            spin_vectors: Some(spins.clone()),
        },
    ];
    let ordinary = runtime.evaluate_f64(&momenta, 2)?;
    let result = runtime.evaluate_correlated_many_f64(&momenta, 2, &requests, &[])?;
    runtime.set_spin_correlation_vectors(&spins)?;
    let default_request = CorrelatedRequest::new("cascade-interference");
    let inherited = runtime.evaluate_correlated_f64(&momenta, 2, &default_request, &[])?;
    assert_eq!(ordinary, runtime.evaluate_f64(&momenta, 2)?);
    runtime.set_spin_correlation_vectors(&[])?;
    let cleared = runtime.evaluate_correlated_f64(&momenta, 2, &default_request, &[])?;
    let close = |a: Complex64, b: Complex64| {
        (a.re - b.re).hypot(a.im - b.im)
            <= 1e-11 * a.re.hypot(a.im).max(b.re.hypot(b.im)).max(1e-100)
    };
    for point in 0..2 {
        assert!(close(inherited[point], result.get(2, point).unwrap()));
        assert!(close(cleared[point], result.get(1, point).unwrap()));
    }
    for request in 0..requests.len() {
        for point in 0..2 {
            let value = result.get(request, point).unwrap();
            println!(
                "VALUE {request} {point} {:.17e} {:.17e}",
                value.re, value.im
            );
        }
    }
    Ok(())
}

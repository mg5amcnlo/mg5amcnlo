// SPDX-License-Identifier: 0BSD

use super::*;

pub(super) fn c64(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

pub(super) fn negate(momentum: [f64; 4]) -> [f64; 4] {
    [-momentum[0], -momentum[1], -momentum[2], -momentum[3]]
}

pub(super) fn fortran_sign(value: f64, sign_source: f64) -> f64 {
    value.abs().copysign(sign_source)
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn complex_zero<T>() -> Complex<T>
where
    T: Real + Clone,
{
    Complex::new(T::new_zero(), T::new_zero())
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn c_generic<T>(re: T, im: T) -> Complex<T> {
    Complex::new(re, im)
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn negate_generic<T>(momentum: &[T; 4]) -> [T; 4]
where
    T: Real + Clone,
{
    [
        -momentum[0].clone(),
        -momentum[1].clone(),
        -momentum[2].clone(),
        -momentum[3].clone(),
    ]
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn is_zero<T: RealLike>(value: &T) -> bool {
    value.to_f64() == 0.0
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn t_min<T>(left: &T, right: &T) -> T
where
    T: RealLike + PartialOrd + Clone,
{
    if left <= right {
        left.clone()
    } else {
        right.clone()
    }
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn t_max<T>(left: &T, right: &T) -> T
where
    T: RealLike + PartialOrd + Clone,
{
    if left >= right {
        left.clone()
    } else {
        right.clone()
    }
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn fortran_sign_generic<T>(value: &T, sign_source: &T) -> T
where
    T: Real + RealLike + Clone,
{
    let magnitude = value.norm();
    if sign_source.to_f64().is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

pub(super) fn ext_quark_dirac(momentum: [f64; 4], helicity: i32) -> [Complex<f64>; 4] {
    let [energy, px, py, pz] = momentum;
    if energy > 0.0 {
        let sqp0p3 = if px == 0.0 && py == 0.0 && pz < 0.0 {
            0.0
        } else {
            (energy + pz).max(0.0).sqrt()
        };
        let chi1 = c64(sqp0p3, 0.0);
        let chi2 = if sqp0p3 == 0.0 {
            c64(-(helicity as f64) * (2.0 * energy).sqrt(), 0.0)
        } else {
            c64(helicity as f64 * px / sqp0p3, -py / sqp0p3)
        };
        if helicity == 1 {
            return [chi1, chi2, c64(0.0, 0.0), c64(0.0, 0.0)];
        }
        return [c64(0.0, 0.0), c64(0.0, 0.0), chi2, chi1];
    }
    let sqp0p3 = if px == 0.0 && py == 0.0 && pz > 0.0 {
        0.0
    } else {
        -(-(energy + pz)).max(0.0).sqrt()
    };
    let chi1 = c64(sqp0p3, 0.0);
    let chi2 = if sqp0p3 == 0.0 {
        c64(-(helicity as f64) * (2.0 * energy.abs()).sqrt(), 0.0)
    } else {
        c64(-(helicity as f64) * (-px) / sqp0p3, py / sqp0p3)
    };
    if -helicity == 1 {
        [chi1, chi2, c64(0.0, 0.0), c64(0.0, 0.0)]
    } else {
        [c64(0.0, 0.0), c64(0.0, 0.0), chi2, chi1]
    }
}

pub(super) fn ext_antiquark_dirac(momentum: [f64; 4], helicity: i32) -> [Complex<f64>; 4] {
    let [energy, px, py, pz] = momentum;
    if energy > 0.0 {
        let sqp0p3 = if px == 0.0 && py == 0.0 && pz < 0.0 {
            0.0
        } else {
            -(energy + pz).max(0.0).sqrt()
        };
        let chi1 = c64(sqp0p3, 0.0);
        let chi2 = if sqp0p3 == 0.0 {
            c64(-(helicity as f64) * (2.0 * energy).sqrt(), 0.0)
        } else {
            c64(-(helicity as f64) * px / sqp0p3, py / sqp0p3)
        };
        if -helicity == 1 {
            return [c64(0.0, 0.0), c64(0.0, 0.0), chi1, chi2];
        }
        return [chi2, chi1, c64(0.0, 0.0), c64(0.0, 0.0)];
    }
    let sqp0p3 = if px == 0.0 && py == 0.0 && pz > 0.0 {
        0.0
    } else {
        (-(energy + pz)).max(0.0).sqrt()
    };
    let chi1 = c64(sqp0p3, 0.0);
    let chi2 = if sqp0p3 == 0.0 {
        c64(-(helicity as f64) * (2.0 * energy.abs()).sqrt(), 0.0)
    } else {
        c64(helicity as f64 * (-px) / sqp0p3, -py / sqp0p3)
    };
    if helicity == 1 {
        [c64(0.0, 0.0), c64(0.0, 0.0), chi1, chi2]
    } else {
        [chi2, chi1, c64(0.0, 0.0), c64(0.0, 0.0)]
    }
}

pub(super) fn ext_quark_dirac_massive(
    momentum: [f64; 4],
    helicity: i32,
    mass: f64,
) -> [Complex<f64>; 4] {
    if mass.abs() < 1.0e-8 {
        return ext_quark_dirac(momentum, helicity);
    }
    let [energy, px, py, pz] = momentum;
    let nsf = if energy > 0.0 { 1 } else { -1 };
    let nh = nsf * helicity;
    let pp = (px * px + py * py + pz * pz).sqrt().abs();
    let omega1 = (energy.abs() + pp).sqrt();
    let omega2 = mass / omega1;
    let omega = [omega1, omega2];
    let sf1 = (1 + nsf + (1 - nsf) * nh) as f64 * 0.5;
    let sf2 = (1 + nsf - (1 - nsf) * nh) as f64 * 0.5;
    let ip = ((3 + nh) / 2 - 1) as usize;
    let im = ((3 - nh) / 2 - 1) as usize;
    let sfomeg = [sf1 * omega[ip], sf2 * omega[im]];
    let (signed_px, signed_py, signed_pz) = if energy > 0.0 {
        (px, py, pz)
    } else {
        (-px, -py, -pz)
    };
    let pp3 = (pp + signed_pz).max(0.0);
    let chi1 = if pp == 0.0 {
        c64(1.0, 0.0)
    } else {
        c64((pp3 * 0.5 / pp).sqrt(), 0.0)
    };
    let chi2 = if pp3 == 0.0 || pp == 0.0 {
        c64(-(nh as f64), 0.0)
    } else {
        let denom = (2.0 * pp * pp3).sqrt();
        c64((nh as f64) * signed_px / denom, -signed_py / denom)
    };
    let chi = [chi1, chi2];
    [
        chi[im] * sfomeg[1],
        chi[ip] * sfomeg[1],
        chi[im] * sfomeg[0],
        chi[ip] * sfomeg[0],
    ]
}

pub(super) fn ext_antiquark_dirac_massive(
    momentum: [f64; 4],
    helicity: i32,
    mass: f64,
) -> [Complex<f64>; 4] {
    if mass.abs() < 1.0e-8 {
        return ext_antiquark_dirac(momentum, helicity);
    }
    let [energy, px, py, pz] = momentum;
    let nsf = if energy > 0.0 { -1 } else { 1 };
    let nh = nsf * helicity;
    let pp = (px * px + py * py + pz * pz).sqrt().abs();
    let omega1 = (energy.abs() + pp).sqrt();
    let omega2 = mass / omega1;
    let omega = [omega1, omega2];
    let sf1 = (1 + nsf + (1 - nsf) * nh) as f64 * 0.5;
    let sf2 = (1 + nsf - (1 - nsf) * nh) as f64 * 0.5;
    let ip = ((3 + nh) / 2 - 1) as usize;
    let im = ((3 - nh) / 2 - 1) as usize;
    let sfomeg = [sf1 * omega[ip], sf2 * omega[im]];
    let (signed_px, signed_py, signed_pz) = if energy > 0.0 {
        (px, py, pz)
    } else {
        (-px, -py, -pz)
    };
    let pp3 = (pp + signed_pz).max(0.0);
    let chi1 = if pp == 0.0 {
        c64(1.0, 0.0)
    } else {
        c64((pp3 * 0.5 / pp).sqrt(), 0.0)
    };
    let chi2 = if pp3 == 0.0 || pp == 0.0 {
        c64(-(nh as f64), 0.0)
    } else {
        let denom = (2.0 * pp * pp3).sqrt();
        c64((nh as f64) * signed_px / denom, signed_py / denom)
    };
    let chi = [chi1, chi2];
    [
        chi[im] * sfomeg[0],
        chi[ip] * sfomeg[0],
        chi[im] * sfomeg[1],
        chi[ip] * sfomeg[1],
    ]
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn ext_quark_dirac_generic<T>(momentum: &[T; 4], helicity: i32) -> [Complex<T>; 4]
where
    T: Real + RealLike + From<f64> + PartialOrd + Clone,
{
    let energy = momentum[0].clone();
    let px = momentum[1].clone();
    let py = momentum[2].clone();
    let pz = momentum[3].clone();
    if energy.to_f64() > 0.0 {
        let sqp0p3 = if is_zero(&px) && is_zero(&py) && pz.to_f64() < 0.0 {
            T::new_zero()
        } else {
            t_max(&(energy.clone() + pz.clone()), &T::new_zero()).sqrt()
        };
        let chi1 = c_generic(sqp0p3.clone(), T::new_zero());
        let chi2 = if is_zero(&sqp0p3) {
            c_generic(
                -energy.from_i64(helicity as i64) * (energy.from_i64(2) * energy.clone()).sqrt(),
                T::new_zero(),
            )
        } else {
            c_generic(
                energy.from_i64(helicity as i64) * px.clone() / sqp0p3.clone(),
                -py.clone() / sqp0p3.clone(),
            )
        };
        if helicity == 1 {
            return [chi1, chi2, complex_zero(), complex_zero()];
        }
        return [complex_zero(), complex_zero(), chi2, chi1];
    }
    let sqp0p3 = if is_zero(&px) && is_zero(&py) && pz.to_f64() > 0.0 {
        T::new_zero()
    } else {
        -t_max(&(-(energy.clone() + pz.clone())), &T::new_zero()).sqrt()
    };
    let chi1 = c_generic(sqp0p3.clone(), T::new_zero());
    let chi2 = if is_zero(&sqp0p3) {
        c_generic(
            -energy.from_i64(helicity as i64) * (energy.from_i64(2) * energy.norm()).sqrt(),
            T::new_zero(),
        )
    } else {
        c_generic(
            -energy.from_i64(helicity as i64) * (-px.clone()) / sqp0p3.clone(),
            py.clone() / sqp0p3.clone(),
        )
    };
    if -helicity == 1 {
        [chi1, chi2, complex_zero(), complex_zero()]
    } else {
        [complex_zero(), complex_zero(), chi2, chi1]
    }
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn ext_antiquark_dirac_generic<T>(momentum: &[T; 4], helicity: i32) -> [Complex<T>; 4]
where
    T: Real + RealLike + From<f64> + PartialOrd + Clone,
{
    let energy = momentum[0].clone();
    let px = momentum[1].clone();
    let py = momentum[2].clone();
    let pz = momentum[3].clone();
    if energy.to_f64() > 0.0 {
        let sqp0p3 = if is_zero(&px) && is_zero(&py) && pz.to_f64() < 0.0 {
            T::new_zero()
        } else {
            -t_max(&(energy.clone() + pz.clone()), &T::new_zero()).sqrt()
        };
        let chi1 = c_generic(sqp0p3.clone(), T::new_zero());
        let chi2 = if is_zero(&sqp0p3) {
            c_generic(
                -energy.from_i64(helicity as i64) * (energy.from_i64(2) * energy.clone()).sqrt(),
                T::new_zero(),
            )
        } else {
            c_generic(
                -energy.from_i64(helicity as i64) * px.clone() / sqp0p3.clone(),
                py.clone() / sqp0p3.clone(),
            )
        };
        if -helicity == 1 {
            return [complex_zero(), complex_zero(), chi1, chi2];
        }
        return [chi2, chi1, complex_zero(), complex_zero()];
    }
    let sqp0p3 = if is_zero(&px) && is_zero(&py) && pz.to_f64() > 0.0 {
        T::new_zero()
    } else {
        t_max(&(-(energy.clone() + pz.clone())), &T::new_zero()).sqrt()
    };
    let chi1 = c_generic(sqp0p3.clone(), T::new_zero());
    let chi2 = if is_zero(&sqp0p3) {
        c_generic(
            -energy.from_i64(helicity as i64) * (energy.from_i64(2) * energy.norm()).sqrt(),
            T::new_zero(),
        )
    } else {
        c_generic(
            energy.from_i64(helicity as i64) * (-px.clone()) / sqp0p3.clone(),
            -py.clone() / sqp0p3.clone(),
        )
    };
    if helicity == 1 {
        [complex_zero(), complex_zero(), chi1, chi2]
    } else {
        [chi2, chi1, complex_zero(), complex_zero()]
    }
}

pub(super) fn ext_massive_vector(
    momentum: [f64; 4],
    helicity: i32,
    mass: f64,
) -> [Complex<f64>; 4] {
    let [energy, px, py, pz] = momentum;
    if energy < 0.0 {
        return ext_massive_vector(negate(momentum), -helicity, mass);
    }
    let sqh = 0.5f64.sqrt();
    let hel = helicity as f64;
    let nsvahl = helicity.abs() as f64;
    let pt2 = px * px + py * py;
    let pp = energy.min((pt2 + pz * pz).sqrt());
    let pt = pp.min(pt2.sqrt());
    let hel0 = 1.0 - hel.abs();
    if pp == 0.0 {
        return [
            c64(0.0, 0.0),
            c64(-hel * sqh, 0.0),
            c64(0.0, nsvahl * sqh),
            c64(hel0, 0.0),
        ];
    }
    let emp = energy / (mass * pp);
    let wf0 = c64(hel0 * pp / mass, 0.0);
    let wf3 = c64(hel0 * pz * emp + hel * pt / pp * sqh, 0.0);
    let (wf1, wf2) = if pt != 0.0 {
        let pzpt = pz / (pp * pt) * sqh * hel;
        (
            c64(hel0 * px * emp - px * pzpt, -nsvahl * py / pt * sqh),
            c64(hel0 * py * emp - py * pzpt, nsvahl * px / pt * sqh),
        )
    } else {
        (
            c64(-hel * sqh, 0.0),
            c64(0.0, nsvahl * fortran_sign(sqh, pz)),
        )
    };
    [wf0, wf1, wf2, wf3]
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn ext_massive_vector_generic<T>(
    momentum: &[T; 4],
    helicity: i32,
    mass: T,
) -> [Complex<T>; 4]
where
    T: Real + RealLike + From<f64> + PartialOrd + Clone,
{
    let energy = momentum[0].clone();
    if energy.to_f64() < 0.0 {
        return ext_massive_vector_generic(&negate_generic(momentum), -helicity, mass);
    }
    let px = momentum[1].clone();
    let py = momentum[2].clone();
    let pz = momentum[3].clone();
    let sqh = (energy.one() / energy.from_i64(2)).sqrt();
    let hel = energy.from_i64(helicity as i64);
    let nsvahl = energy.from_i64(helicity.abs() as i64);
    let pt2 = px.clone() * px.clone() + py.clone() * py.clone();
    let pp = t_min(&energy, &(pt2.clone() + pz.clone() * pz.clone()).sqrt());
    let pt = t_min(&pp, &pt2.sqrt());
    let hel0 = if helicity == 0 {
        energy.one()
    } else {
        T::new_zero()
    };
    if is_zero(&pp) {
        return [
            complex_zero(),
            c_generic(-hel.clone() * sqh.clone(), T::new_zero()),
            c_generic(T::new_zero(), nsvahl.clone() * sqh.clone()),
            c_generic(hel0, T::new_zero()),
        ];
    }
    let emp = energy.clone() / (mass.clone() * pp.clone());
    let wf0 = c_generic(hel0.clone() * pp.clone() / mass, T::new_zero());
    let wf3 = c_generic(
        hel0.clone() * pz.clone() * emp.clone()
            + hel.clone() * pt.clone() / pp.clone() * sqh.clone(),
        T::new_zero(),
    );
    let (wf1, wf2) = if !is_zero(&pt) {
        let pzpt = pz.clone() / (pp.clone() * pt.clone()) * sqh.clone() * hel.clone();
        (
            c_generic(
                hel0.clone() * px.clone() * emp.clone() - px.clone() * pzpt.clone(),
                -nsvahl.clone() * py.clone() / pt.clone() * sqh.clone(),
            ),
            c_generic(
                hel0 * py.clone() * emp - py.clone() * pzpt,
                nsvahl * px.clone() / pt * sqh,
            ),
        )
    } else {
        (
            c_generic(-hel * sqh.clone(), T::new_zero()),
            c_generic(T::new_zero(), nsvahl * fortran_sign_generic(&sqh, &pz)),
        )
    };
    [wf0, wf1, wf2, wf3]
}

pub(super) fn spin2_outer(
    left: &[Complex<f64>; 4],
    right: &[Complex<f64>; 4],
) -> [Complex<f64>; 16] {
    std::array::from_fn(|index| left[index / 4] * right[index % 4])
}

pub(super) fn ext_spin2(
    momentum: [f64; 4],
    helicity: i32,
    mass: f64,
) -> RusticolResult<[Complex<f64>; 16]> {
    if mass == 0.0 {
        if ![-2, 2].contains(&helicity) {
            return Err(RusticolError::invalid_argument(
                "massless spin-2 sources only support helicities -2 and 2",
            ));
        }
        let vector = ext_gluon(momentum, helicity / 2);
        return Ok(spin2_outer(&vector, &vector));
    }
    if ![-2, -1, 0, 1, 2].contains(&helicity) {
        return Err(RusticolError::invalid_argument(format!(
            "unsupported massive spin-2 helicity {helicity}"
        )));
    }
    let plus = ext_massive_vector(momentum, 1, mass);
    let minus = ext_massive_vector(momentum, -1, mass);
    let longitudinal = ext_massive_vector(momentum, 0, mass);
    let plus_plus = spin2_outer(&plus, &plus);
    let minus_minus = spin2_outer(&minus, &minus);
    if helicity == 2 {
        return Ok(plus_plus);
    }
    if helicity == -2 {
        return Ok(minus_minus);
    }
    let inverse_sqrt_two = 1.0 / 2.0f64.sqrt();
    if helicity == 1 {
        let first = spin2_outer(&plus, &longitudinal);
        let second = spin2_outer(&longitudinal, &plus);
        return Ok(std::array::from_fn(|index| {
            (first[index] + second[index]) * inverse_sqrt_two
        }));
    }
    if helicity == -1 {
        let first = spin2_outer(&minus, &longitudinal);
        let second = spin2_outer(&longitudinal, &minus);
        return Ok(std::array::from_fn(|index| {
            (first[index] + second[index]) * inverse_sqrt_two
        }));
    }
    let plus_minus = spin2_outer(&plus, &minus);
    let minus_plus = spin2_outer(&minus, &plus);
    let zero_zero = spin2_outer(&longitudinal, &longitudinal);
    let inverse_sqrt_six = 1.0 / 6.0f64.sqrt();
    Ok(std::array::from_fn(|index| {
        (plus_minus[index] + minus_plus[index] + c64(2.0, 0.0) * zero_zero[index])
            * inverse_sqrt_six
    }))
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn spin2_outer_generic<T>(
    left: &[Complex<T>; 4],
    right: &[Complex<T>; 4],
) -> [Complex<T>; 16]
where
    T: Real + RealLike + From<f64> + Clone,
{
    std::array::from_fn(|index| left[index / 4].clone() * right[index % 4].clone())
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn ext_spin2_generic<T>(
    momentum: &[T; 4],
    helicity: i32,
    mass: T,
) -> RusticolResult<[Complex<T>; 16]>
where
    T: Real + RealLike + From<f64> + PartialOrd + Clone,
{
    if is_zero(&mass) {
        if ![-2, 2].contains(&helicity) {
            return Err(RusticolError::invalid_argument(
                "massless spin-2 sources only support helicities -2 and 2",
            ));
        }
        let vector = ext_gluon_generic(momentum, helicity / 2);
        return Ok(spin2_outer_generic(&vector, &vector));
    }
    if ![-2, -1, 0, 1, 2].contains(&helicity) {
        return Err(RusticolError::invalid_argument(format!(
            "unsupported massive spin-2 helicity {helicity}"
        )));
    }
    let plus = ext_massive_vector_generic(momentum, 1, mass.clone());
    let minus = ext_massive_vector_generic(momentum, -1, mass.clone());
    let longitudinal = ext_massive_vector_generic(momentum, 0, mass.clone());
    if helicity == 2 {
        return Ok(spin2_outer_generic(&plus, &plus));
    }
    if helicity == -2 {
        return Ok(spin2_outer_generic(&minus, &minus));
    }
    let inverse_sqrt_two = mass.one() / mass.from_i64(2).sqrt();
    let weight_two = c_generic(inverse_sqrt_two, T::new_zero());
    if helicity == 1 {
        let first = spin2_outer_generic(&plus, &longitudinal);
        let second = spin2_outer_generic(&longitudinal, &plus);
        return Ok(std::array::from_fn(|index| {
            (first[index].clone() + second[index].clone()) * weight_two.clone()
        }));
    }
    if helicity == -1 {
        let first = spin2_outer_generic(&minus, &longitudinal);
        let second = spin2_outer_generic(&longitudinal, &minus);
        return Ok(std::array::from_fn(|index| {
            (first[index].clone() + second[index].clone()) * weight_two.clone()
        }));
    }
    let plus_minus = spin2_outer_generic(&plus, &minus);
    let minus_plus = spin2_outer_generic(&minus, &plus);
    let zero_zero = spin2_outer_generic(&longitudinal, &longitudinal);
    let two = c_generic(mass.from_i64(2), T::new_zero());
    let inverse_sqrt_six = c_generic(mass.one() / mass.from_i64(6).sqrt(), T::new_zero());
    Ok(std::array::from_fn(|index| {
        (plus_minus[index].clone()
            + minus_plus[index].clone()
            + two.clone() * zero_zero[index].clone())
            * inverse_sqrt_six.clone()
    }))
}

pub(super) fn ext_gluon(momentum: [f64; 4], helicity: i32) -> [Complex<f64>; 4] {
    let [energy, px, py, pz] = momentum;
    let sqh = 0.5f64.sqrt();
    if energy > 0.0 {
        let hel = helicity as f64;
        let pp = energy;
        let pt = (px * px + py * py).sqrt();
        let wf3 = c64(hel * pt / pp * sqh, 0.0);
        let (wf1, wf2) = if pt != 0.0 {
            let pzpt = pz / (pp * pt) * sqh * hel;
            (
                c64(-px * pzpt, -py / pt * sqh),
                c64(-py * pzpt, px / pt * sqh),
            )
        } else {
            (c64(-hel * sqh, 0.0), c64(0.0, fortran_sign(sqh, pz)))
        };
        return [c64(0.0, 0.0), wf1, wf2, wf3];
    }
    let hel = -helicity as f64;
    let pp = -energy;
    let pt = (px * px + py * py).sqrt();
    let wf3 = c64(hel * pt / pp * sqh, 0.0);
    let (wf1, wf2) = if pt != 0.0 {
        let pzpt = -pz / (pp * pt) * sqh * hel;
        (
            c64(px * pzpt, py / pt * sqh),
            c64(py * pzpt, -px / pt * sqh),
        )
    } else {
        (c64(-hel * sqh, 0.0), c64(0.0, -fortran_sign(sqh, pz)))
    };
    [c64(0.0, 0.0), wf1, wf2, wf3]
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn ext_gluon_generic<T>(momentum: &[T; 4], helicity: i32) -> [Complex<T>; 4]
where
    T: Real + RealLike + From<f64> + Clone,
{
    let energy = momentum[0].clone();
    let px = momentum[1].clone();
    let py = momentum[2].clone();
    let pz = momentum[3].clone();
    let sqh = (energy.one() / energy.from_i64(2)).sqrt();
    if energy.to_f64() > 0.0 {
        let hel = energy.from_i64(helicity as i64);
        let pp = energy;
        let pt = (px.clone() * px.clone() + py.clone() * py.clone()).sqrt();
        let wf3 = c_generic(
            hel.clone() * pt.clone() / pp.clone() * sqh.clone(),
            T::new_zero(),
        );
        let (wf1, wf2) = if !is_zero(&pt) {
            let pzpt = pz.clone() / (pp.clone() * pt.clone()) * sqh.clone() * hel.clone();
            (
                c_generic(
                    -px.clone() * pzpt.clone(),
                    -py.clone() / pt.clone() * sqh.clone(),
                ),
                c_generic(-py.clone() * pzpt, px.clone() / pt * sqh),
            )
        } else {
            (
                c_generic(-hel * sqh.clone(), T::new_zero()),
                c_generic(T::new_zero(), fortran_sign_generic(&sqh, &pz)),
            )
        };
        return [complex_zero(), wf1, wf2, wf3];
    }
    let hel = energy.from_i64(-(helicity as i64));
    let pp = -energy;
    let pt = (px.clone() * px.clone() + py.clone() * py.clone()).sqrt();
    let wf3 = c_generic(
        hel.clone() * pt.clone() / pp.clone() * sqh.clone(),
        T::new_zero(),
    );
    let (wf1, wf2) = if !is_zero(&pt) {
        let pzpt = -pz.clone() / (pp.clone() * pt.clone()) * sqh.clone() * hel.clone();
        (
            c_generic(
                px.clone() * pzpt.clone(),
                py.clone() / pt.clone() * sqh.clone(),
            ),
            c_generic(py.clone() * pzpt, -px.clone() / pt * sqh),
        )
    } else {
        (
            c_generic(-hel * sqh.clone(), T::new_zero()),
            c_generic(T::new_zero(), -fortran_sign_generic(&sqh, &pz)),
        )
    };
    [complex_zero(), wf1, wf2, wf3]
}

pub(super) fn ext_quark_weyl_array(
    momentum: [f64; 4],
    helicity: i32,
    chirality: i32,
) -> [Complex<f64>; 2] {
    let [energy, px, py, pz] = momentum;
    if energy > 0.0 {
        let sqp0p3 = if px == 0.0 && py == 0.0 && pz < 0.0 {
            0.0
        } else {
            (energy + pz).max(0.0).sqrt()
        };
        let chi1 = c64(sqp0p3, 0.0);
        let chi2 = if sqp0p3 == 0.0 {
            c64(-(helicity as f64) * (2.0 * energy).sqrt(), 0.0)
        } else {
            c64(helicity as f64 * px / sqp0p3, -py / sqp0p3)
        };
        if helicity == 1 && chirality == 1 {
            return [chi1, chi2];
        }
        if helicity == -1 && chirality == -1 {
            return [chi2, chi1];
        }
        return [c64(0.0, 0.0), c64(0.0, 0.0)];
    }
    let sqp0p3 = if px == 0.0 && py == 0.0 && pz > 0.0 {
        0.0
    } else {
        -(-(energy + pz)).max(0.0).sqrt()
    };
    let chi1 = c64(sqp0p3, 0.0);
    let chi2 = if sqp0p3 == 0.0 {
        c64(-(helicity as f64) * (2.0 * energy.abs()).sqrt(), 0.0)
    } else {
        c64(-(helicity as f64) * (-px) / sqp0p3, py / sqp0p3)
    };
    if helicity == -1 && chirality == 1 {
        [chi1, chi2]
    } else if helicity == 1 && chirality == -1 {
        [chi2, chi1]
    } else {
        [c64(0.0, 0.0), c64(0.0, 0.0)]
    }
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn ext_quark_weyl_generic<T>(
    momentum: &[T; 4],
    helicity: i32,
    chirality: i32,
) -> Vec<Complex<T>>
where
    T: Real + RealLike + From<f64> + PartialOrd + Clone,
{
    let energy = momentum[0].clone();
    let px = momentum[1].clone();
    let py = momentum[2].clone();
    let pz = momentum[3].clone();
    if energy.to_f64() > 0.0 {
        let sqp0p3 = if is_zero(&px) && is_zero(&py) && pz.to_f64() < 0.0 {
            T::new_zero()
        } else {
            t_max(&(energy.clone() + pz.clone()), &T::new_zero()).sqrt()
        };
        let chi1 = c_generic(sqp0p3.clone(), T::new_zero());
        let chi2 = if is_zero(&sqp0p3) {
            c_generic(
                -energy.from_i64(helicity as i64) * (energy.from_i64(2) * energy.clone()).sqrt(),
                T::new_zero(),
            )
        } else {
            c_generic(
                energy.from_i64(helicity as i64) * px.clone() / sqp0p3.clone(),
                -py.clone() / sqp0p3.clone(),
            )
        };
        if helicity == 1 && chirality == 1 {
            return vec![chi1, chi2];
        }
        if helicity == -1 && chirality == -1 {
            return vec![chi2, chi1];
        }
        return vec![complex_zero(), complex_zero()];
    }
    let sqp0p3 = if is_zero(&px) && is_zero(&py) && pz.to_f64() > 0.0 {
        T::new_zero()
    } else {
        -t_max(&(-(energy.clone() + pz.clone())), &T::new_zero()).sqrt()
    };
    let chi1 = c_generic(sqp0p3.clone(), T::new_zero());
    let chi2 = if is_zero(&sqp0p3) {
        c_generic(
            -energy.from_i64(helicity as i64) * (energy.from_i64(2) * energy.norm()).sqrt(),
            T::new_zero(),
        )
    } else {
        c_generic(
            -energy.from_i64(helicity as i64) * (-px.clone()) / sqp0p3.clone(),
            py.clone() / sqp0p3.clone(),
        )
    };
    if helicity == -1 && chirality == 1 {
        vec![chi1, chi2]
    } else if helicity == 1 && chirality == -1 {
        vec![chi2, chi1]
    } else {
        vec![complex_zero(), complex_zero()]
    }
}

pub(super) fn ext_antiquark_weyl_array(
    momentum: [f64; 4],
    helicity: i32,
    chirality: i32,
) -> [Complex<f64>; 2] {
    let [energy, px, py, pz] = momentum;
    if energy > 0.0 {
        let sqp0p3 = if px == 0.0 && py == 0.0 && pz < 0.0 {
            0.0
        } else {
            -(energy + pz).max(0.0).sqrt()
        };
        let chi1 = c64(sqp0p3, 0.0);
        let chi2 = if sqp0p3 == 0.0 {
            c64(-(helicity as f64) * (2.0 * energy).sqrt(), 0.0)
        } else {
            c64(-(helicity as f64) * px / sqp0p3, py / sqp0p3)
        };
        if helicity == 1 && chirality == 1 {
            return [chi2, chi1];
        }
        if helicity == -1 && chirality == -1 {
            return [chi1, chi2];
        }
        return [c64(0.0, 0.0), c64(0.0, 0.0)];
    }
    let sqp0p3 = if px == 0.0 && py == 0.0 && pz > 0.0 {
        0.0
    } else {
        (-(energy + pz)).max(0.0).sqrt()
    };
    let chi1 = c64(sqp0p3, 0.0);
    let chi2 = if sqp0p3 == 0.0 {
        c64(-(helicity as f64) * (2.0 * energy.abs()).sqrt(), 0.0)
    } else {
        c64(helicity as f64 * (-px) / sqp0p3, -py / sqp0p3)
    };
    if helicity == -1 && chirality == 1 {
        [chi2, chi1]
    } else if helicity == 1 && chirality == -1 {
        [chi1, chi2]
    } else {
        [c64(0.0, 0.0), c64(0.0, 0.0)]
    }
}

#[cfg(feature = "symbolica-runtime")]
pub(super) fn ext_antiquark_weyl_generic<T>(
    momentum: &[T; 4],
    helicity: i32,
    chirality: i32,
) -> Vec<Complex<T>>
where
    T: Real + RealLike + From<f64> + PartialOrd + Clone,
{
    let energy = momentum[0].clone();
    let px = momentum[1].clone();
    let py = momentum[2].clone();
    let pz = momentum[3].clone();
    if energy.to_f64() > 0.0 {
        let sqp0p3 = if is_zero(&px) && is_zero(&py) && pz.to_f64() < 0.0 {
            T::new_zero()
        } else {
            -t_max(&(energy.clone() + pz.clone()), &T::new_zero()).sqrt()
        };
        let chi1 = c_generic(sqp0p3.clone(), T::new_zero());
        let chi2 = if is_zero(&sqp0p3) {
            c_generic(
                -energy.from_i64(helicity as i64) * (energy.from_i64(2) * energy.clone()).sqrt(),
                T::new_zero(),
            )
        } else {
            c_generic(
                -energy.from_i64(helicity as i64) * px.clone() / sqp0p3.clone(),
                py.clone() / sqp0p3.clone(),
            )
        };
        if helicity == 1 && chirality == 1 {
            return vec![chi2, chi1];
        }
        if helicity == -1 && chirality == -1 {
            return vec![chi1, chi2];
        }
        return vec![complex_zero(), complex_zero()];
    }
    let sqp0p3 = if is_zero(&px) && is_zero(&py) && pz.to_f64() > 0.0 {
        T::new_zero()
    } else {
        t_max(&(-(energy.clone() + pz.clone())), &T::new_zero()).sqrt()
    };
    let chi1 = c_generic(sqp0p3.clone(), T::new_zero());
    let chi2 = if is_zero(&sqp0p3) {
        c_generic(
            -energy.from_i64(helicity as i64) * (energy.from_i64(2) * energy.norm()).sqrt(),
            T::new_zero(),
        )
    } else {
        c_generic(
            energy.from_i64(helicity as i64) * (-px.clone()) / sqp0p3.clone(),
            -py.clone() / sqp0p3.clone(),
        )
    };
    if helicity == -1 && chirality == 1 {
        vec![chi2, chi1]
    } else if helicity == 1 && chirality == -1 {
        vec![chi1, chi2]
    } else {
        vec![complex_zero(), complex_zero()]
    }
}

#[cfg(test)]
mod tests {
    use super::{Complex, ext_massive_vector, negate};

    fn assert_complex_close(left: Complex<f64>, right: Complex<f64>) {
        assert!((left.re - right.re).abs() < 1.0e-13);
        assert!((left.im - right.im).abs() < 1.0e-13);
    }

    #[test]
    fn massive_vector_negative_energy_obeys_crossing_convention() {
        let momentum = [13.0, 4.0, 8.0, 8.0];
        for helicity in [-1, 0, 1] {
            let outgoing = ext_massive_vector(momentum, helicity, 5.0);
            let incoming = ext_massive_vector(negate(momentum), -helicity, 5.0);
            for (left, right) in incoming.into_iter().zip(outgoing) {
                assert_complex_close(left, right);
            }
        }
    }

    #[test]
    fn massive_vector_is_transverse_for_both_energy_signs() {
        let momentum = [13.0, 4.0, 8.0, 8.0];
        for helicity in [-1, 0, 1] {
            for (candidate, source_helicity) in
                [(momentum, helicity), (negate(momentum), -helicity)]
            {
                let wave = ext_massive_vector(candidate, source_helicity, 5.0);
                let contraction_re = candidate[0] * wave[0].re
                    - candidate[1] * wave[1].re
                    - candidate[2] * wave[2].re
                    - candidate[3] * wave[3].re;
                let contraction_im = candidate[0] * wave[0].im
                    - candidate[1] * wave[1].im
                    - candidate[2] * wave[2].im
                    - candidate[3] * wave[3].im;
                assert!(
                    contraction_re.hypot(contraction_im) < 1.0e-12,
                    "({contraction_re}, {contraction_im})"
                );
            }
        }
    }
}

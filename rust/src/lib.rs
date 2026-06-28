use nalgebra::{Matrix3, Vector3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const NP: usize = 3; // every model has params [S0, T, offset]

#[derive(Clone, Copy)]
enum Model {
    MonoExp,
    AronenT1rho,
    AronenT2,
    Rausch,
}

#[derive(Clone, Copy)]
struct Seq {
    tr: f64,
    t1: f64,
    alpha: f64,
    te: f64,
    t2star: f64,
}

#[inline]
fn eval(model: Model, seq: &Seq, x: f64, p: &[f64; NP]) -> f64 {
    let (s0, t, offset) = (p[0], p[1], p[2]);
    match model {
        Model::MonoExp => s0 * (-x / t).exp() + offset,
        Model::Rausch => {
            let tau = seq.tr - x;
            let num = (1.0 - (-tau / seq.t1).exp()) * (-x / t).exp();
            let den = 1.0 - seq.alpha.cos() * (-x / t).exp() * (-tau / seq.t1).exp();
            s0 * seq.alpha.sin() * num / den + offset
        }
        Model::AronenT1rho | Model::AronenT2 => {
            let tau = seq.tr - x;
            let num = s0
                * (-x / t).exp()
                * (1.0 - (-tau / seq.t1).exp())
                * seq.alpha.sin()
                * (-seq.te / seq.t2star).exp();
            let den = 1.0 - seq.alpha.cos() * (-tau / seq.t1).exp() * (-x / t).exp();
            num / den + offset
        }
    }
}

#[inline]
fn clamp(p: &mut [f64; NP], lo: &[f64; NP], hi: &[f64; NP]) {
    for j in 0..NP {
        if p[j] < lo[j] {
            p[j] = lo[j];
        } else if p[j] > hi[j] {
            p[j] = hi[j];
        }
    }
}

fn sse(model: Model, seq: &Seq, x: &[f64], y: &[f64], p: &[f64; NP]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(&xi, &yi)| {
            let r = eval(model, seq, xi, p) - yi;
            r * r
        })
        .sum()
}

/// Log-linear seed for [S0, T, offset], ignoring offset (offset seed = 0).
fn loglinear_seed(x: &[f64], y: &[f64], lo: &[f64; NP], hi: &[f64; NP]) -> [f64; NP] {
    let mut sx = 0.0;
    let mut sxx = 0.0;
    let mut sl = 0.0;
    let mut sxl = 0.0;
    let mut n = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        if yi > 1e-10 {
            let li = yi.ln();
            sx += xi;
            sxx += xi * xi;
            sl += li;
            sxl += xi * li;
            n += 1.0;
        }
    }
    let mut p = [hi[0].min(1.0).max(lo[0]), 0.5 * (lo[1] + hi[1]), 0.0];
    let denom = n * sxx - sx * sx;
    if n >= 2.0 && denom.abs() > 1e-12 {
        let slope = (n * sxl - sx * sl) / denom;
        let intercept = (sl - slope * sx) / n;
        let s0 = intercept.exp();
        if slope < -1e-12 {
            p[1] = -1.0 / slope;
        }
        p[0] = s0;
    }
    clamp(&mut p, lo, hi);
    p
}

/// Levenberg-Marquardt with numeric forward-difference Jacobian and box bounds.
fn fit_pixel(
    model: Model,
    seq: &Seq,
    x: &[f64],
    y_raw: &[f64],
    lo: &[f64; NP],
    hi: &[f64; NP],
    normalize: bool,
    max_iter: usize,
) -> ([f64; NP], f64) {
    // Normalize per pixel; S0/offset bounds are interpreted in this space.
    let ymax = y_raw.iter().cloned().fold(0.0_f64, f64::max);
    let scale = if normalize && ymax > 0.0 { ymax } else { 1.0 };
    let y: Vec<f64> = y_raw.iter().map(|&v| v / scale).collect();

    let mut p = loglinear_seed(x, &y, lo, hi);
    let mut cost = sse(model, seq, x, &y, &p);
    let mut lambda = 1e-3;

    for _ in 0..max_iter {
        // Build J^T J and J^T r with numeric Jacobian.
        let mut jtj = Matrix3::<f64>::zeros();
        let mut jtr = Vector3::<f64>::zeros();
        for (&xi, &yi) in x.iter().zip(&y) {
            let f0 = eval(model, seq, xi, &p);
            let ri = f0 - yi;
            let mut grad = [0.0; NP];
            for j in 0..NP {
                let h = 1e-6 * p[j].abs().max(1e-6);
                let mut pp = p;
                pp[j] += h;
                grad[j] = (eval(model, seq, xi, &pp) - f0) / h;
            }
            for a in 0..NP {
                jtr[a] += grad[a] * ri;
                for b in 0..NP {
                    jtj[(a, b)] += grad[a] * grad[b];
                }
            }
        }

        // Damped normal equations: (J^T J + lambda*diag) delta = -J^T r
        let mut accepted = false;
        for _ in 0..12 {
            let mut a = jtj;
            for d in 0..NP {
                a[(d, d)] += lambda * jtj[(d, d)].max(1e-12);
            }
            let delta = match a.lu().solve(&(-jtr)) {
                Some(d) => d,
                None => {
                    lambda *= 4.0;
                    continue;
                }
            };
            let mut cand = [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]];
            clamp(&mut cand, lo, hi);
            let new_cost = sse(model, seq, x, &y, &cand);
            if new_cost < cost {
                let improve = cost - new_cost;
                p = cand;
                cost = new_cost;
                lambda = (lambda * 0.5).max(1e-12);
                accepted = true;
                if improve < 1e-12 * (1.0 + cost) {
                    accepted = false; // converged, stop outer loop
                }
                break;
            } else {
                lambda *= 4.0;
                if lambda > 1e12 {
                    break;
                }
            }
        }
        if !accepted {
            break;
        }
    }

    // R^2 in the fitted (possibly normalized) space.
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|&v| (v - mean) * (v - mean)).sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - cost / ss_tot
    } else {
        0.0
    };

    // Undo normalization: S0 and offset scale with signal, T does not.
    p[0] *= scale;
    p[2] *= scale;
    (p, r2)
}

fn model_from_str(name: &str) -> PyResult<Model> {
    match name {
        "mono_exp" => Ok(Model::MonoExp),
        "aronen_t1rho" => Ok(Model::AronenT1rho),
        "aronen_t2" => Ok(Model::AronenT2),
        "rausch" => Ok(Model::Rausch),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown model '{other}'"
        ))),
    }
}

/// Fit a flat stack of pixel signals. Returns (params (n_pixels, 3), r2 (n_pixels,)).
#[pyfunction]
#[pyo3(signature = (signals, x, lower, upper, model, tr=0.0, t1=0.0, alpha=0.0, te=0.0, t2star=0.0, normalize=false, max_iter=100))]
#[allow(clippy::too_many_arguments)]
fn fit_volume<'py>(
    py: Python<'py>,
    signals: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    lower: PyReadonlyArray2<'py, f64>,
    upper: PyReadonlyArray2<'py, f64>,
    model: &str,
    tr: f64,
    t1: f64,
    alpha: f64,
    te: f64,
    t2star: f64,
    normalize: bool,
    max_iter: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let model = model_from_str(model)?;
    let seq = Seq {
        tr,
        t1,
        alpha,
        te,
        t2star,
    };

    let sig = signals.as_array();
    let xs = x.as_array();
    let lo = lower.as_array();
    let hi = upper.as_array();

    let n = sig.nrows();
    let m = sig.ncols();
    if xs.len() != m {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x length does not match signal echo count",
        ));
    }
    if lo.nrows() != n || hi.nrows() != n || lo.ncols() != NP || hi.ncols() != NP {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lower/upper must have shape (n_pixels, 3)",
        ));
    }

    let xv: Vec<f64> = xs.iter().cloned().collect();
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| sig.row(i).iter().cloned().collect())
        .collect();
    let los: Vec<[f64; NP]> = (0..n)
        .map(|i| [lo[(i, 0)], lo[(i, 1)], lo[(i, 2)]])
        .collect();
    let his: Vec<[f64; NP]> = (0..n)
        .map(|i| [hi[(i, 0)], hi[(i, 1)], hi[(i, 2)]])
        .collect();

    let results: Vec<([f64; NP], f64)> = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                fit_pixel(
                    model, &seq, &xv, &rows[i], &los[i], &his[i], normalize, max_iter,
                )
            })
            .collect()
    });

    let mut params = vec![0.0_f64; n * NP];
    let mut r2 = vec![0.0_f64; n];
    for (i, (p, r)) in results.iter().enumerate() {
        params[i * NP] = p[0];
        params[i * NP + 1] = p[1];
        params[i * NP + 2] = p[2];
        r2[i] = *r;
    }

    let params_arr = PyArray1::from_vec_bound(py, params).reshape([n, NP])?;
    let r2_arr = r2.into_pyarray_bound(py);
    Ok((params_arr, r2_arr))
}

#[pymodule]
fn bmri_fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_volume, m)?)?;
    Ok(())
}

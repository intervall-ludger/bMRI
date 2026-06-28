use nalgebra::{DMatrix, DVector};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

mod expr;

/// Hard upper bound for parameter count. All built-in models stay within
/// this. The actual count per model is given by `Model::n_params()`.
const MAX_NP: usize = 4;

#[derive(Clone, Copy, Debug)]
enum Model {
    /// S(x) = S0 * exp(-x/T) + C — T2, T2*, T1rho mono-exponential decay
    MonoExp,
    /// Aronen T2-prep with finite TR (sequence parameters required)
    AronenT2,
    /// Aronen T1rho with finite TR (sequence parameters required)
    AronenT1rho,
    /// Rausch T1rho variant
    Rausch,
    /// DWI mono-exp: S(b) = S0 * exp(-b * D) + C  (D in mm^2/s)
    DwiMonoExp,
    /// DWI Kurtosis: S(b) = S0 * exp(-b*D + b^2 * D^2 * K / 6)
    DwiKurtosis,
    /// IVIM bi-exponential: S(b) = S0 * (f * exp(-b*Dstar) + (1-f) * exp(-b*D))
    DwiIvim,
    /// T2* with two compartments: S(TE) = S0 * (f * exp(-TE/Ts) + (1-f) * exp(-TE/Tl))
    T2starBiExp,
    /// Stretched exponential: S(x) = S0 * exp(-(x/T)^alpha)
    StretchedExp,
    /// User-defined expression parsed at call time.
    Expression,
}

impl Model {
    fn n_params(&self) -> usize {
        match self {
            Model::MonoExp
            | Model::AronenT2
            | Model::AronenT1rho
            | Model::Rausch
            | Model::DwiMonoExp
            | Model::DwiKurtosis
            | Model::StretchedExp => 3,
            Model::DwiIvim | Model::T2starBiExp => 4,
            Model::Expression => 0, // overridden by parsed expression
        }
    }
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
fn eval_builtin(model: Model, seq: &Seq, x: f64, p: &[f64]) -> f64 {
    match model {
        Model::MonoExp => {
            let (s0, t, offset) = (p[0], p[1], p[2]);
            s0 * (-x / t).exp() + offset
        }
        Model::Rausch => {
            let (s0, t, offset) = (p[0], p[1], p[2]);
            let tau = seq.tr - x;
            let num = (1.0 - (-tau / seq.t1).exp()) * (-x / t).exp();
            let den = 1.0 - seq.alpha.cos() * (-x / t).exp() * (-tau / seq.t1).exp();
            s0 * seq.alpha.sin() * num / den + offset
        }
        Model::AronenT1rho | Model::AronenT2 => {
            let (s0, t, offset) = (p[0], p[1], p[2]);
            let tau = seq.tr - x;
            let num = s0
                * (-x / t).exp()
                * (1.0 - (-tau / seq.t1).exp())
                * seq.alpha.sin()
                * (-seq.te / seq.t2star).exp();
            let den = 1.0 - seq.alpha.cos() * (-tau / seq.t1).exp() * (-x / t).exp();
            num / den + offset
        }
        Model::DwiMonoExp => {
            let (s0, d, offset) = (p[0], p[1], p[2]);
            s0 * (-x * d).exp() + offset
        }
        Model::DwiKurtosis => {
            let (s0, d, k) = (p[0], p[1], p[2]);
            let bd = x * d;
            s0 * (-bd + bd * bd * k / 6.0).exp()
        }
        Model::DwiIvim => {
            // p = [S0, D, D_star, f]
            let (s0, d, d_star, f) = (p[0], p[1], p[2], p[3]);
            let ff = f.clamp(0.0, 1.0);
            s0 * (ff * (-x * d_star).exp() + (1.0 - ff) * (-x * d).exp())
        }
        Model::T2starBiExp => {
            // p = [S0, T_short, T_long, f]
            let (s0, ts, tl, f) = (p[0], p[1], p[2], p[3]);
            let ff = f.clamp(0.0, 1.0);
            s0 * (ff * (-x / ts).exp() + (1.0 - ff) * (-x / tl).exp())
        }
        Model::StretchedExp => {
            let (s0, t, a) = (p[0], p[1], p[2]);
            if x <= 0.0 || t <= 0.0 {
                s0
            } else {
                s0 * (-((x / t).powf(a))).exp()
            }
        }
        Model::Expression => unreachable!("expression handled separately"),
    }
}

#[inline]
fn clamp(p: &mut [f64], lo: &[f64], hi: &[f64], np: usize) {
    for j in 0..np {
        if p[j] < lo[j] {
            p[j] = lo[j];
        } else if p[j] > hi[j] {
            p[j] = hi[j];
        }
    }
}

fn sse_builtin(model: Model, seq: &Seq, x: &[f64], y: &[f64], p: &[f64]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(&xi, &yi)| {
            let r = eval_builtin(model, seq, xi, p) - yi;
            r * r
        })
        .sum()
}

/// Log-linear seed for mono-exp shape [S0, T, ...]. Only initialises the first
/// two parameters; the rest are left at the midpoint of their bounds.
fn loglinear_seed(x: &[f64], y: &[f64], lo: &[f64], hi: &[f64], np: usize) -> [f64; MAX_NP] {
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
    let mut p = [0.0_f64; MAX_NP];
    for j in 0..np {
        p[j] = 0.5 * (lo[j] + hi[j]);
    }
    p[0] = hi[0].min(1.0).max(lo[0]);
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
    clamp(&mut p, lo, hi, np);
    p
}

/// Generic Levenberg-Marquardt with numeric Jacobian. Works for any
/// closure that maps (x, params) -> signal and any number of parameters
/// up to MAX_NP.
fn fit_pixel_lm<F>(
    eval_fn: &F,
    x: &[f64],
    y_raw: &[f64],
    lo: &[f64],
    hi: &[f64],
    np: usize,
    seed: [f64; MAX_NP],
    normalize: bool,
    max_iter: usize,
) -> ([f64; MAX_NP], f64)
where
    F: Fn(f64, &[f64]) -> f64,
{
    let ymax = y_raw.iter().cloned().fold(0.0_f64, f64::max);
    let scale = if normalize && ymax > 0.0 { ymax } else { 1.0 };
    let y: Vec<f64> = y_raw.iter().map(|&v| v / scale).collect();

    let mut p = seed;
    let sse = |pp: &[f64; MAX_NP]| -> f64 {
        x.iter()
            .zip(&y)
            .map(|(&xi, &yi)| {
                let r = eval_fn(xi, &pp[..np]) - yi;
                r * r
            })
            .sum()
    };
    let mut cost = sse(&p);
    let mut lambda = 1e-3;

    for _ in 0..max_iter {
        let mut jtj = DMatrix::<f64>::zeros(np, np);
        let mut jtr = DVector::<f64>::zeros(np);
        for (&xi, &yi) in x.iter().zip(&y) {
            let f0 = eval_fn(xi, &p[..np]);
            let ri = f0 - yi;
            let mut grad = [0.0_f64; MAX_NP];
            for j in 0..np {
                let h = 1e-6 * p[j].abs().max(1e-6);
                let mut pp = p;
                pp[j] += h;
                grad[j] = (eval_fn(xi, &pp[..np]) - f0) / h;
            }
            for a in 0..np {
                jtr[a] += grad[a] * ri;
                for b in 0..np {
                    jtj[(a, b)] += grad[a] * grad[b];
                }
            }
        }

        let mut accepted = false;
        for _ in 0..12 {
            let mut a = jtj.clone();
            for d in 0..np {
                a[(d, d)] += lambda * jtj[(d, d)].max(1e-12);
            }
            let delta = match a.clone().lu().solve(&(-&jtr)) {
                Some(d) => d,
                None => {
                    lambda *= 4.0;
                    continue;
                }
            };
            let mut cand = p;
            for j in 0..np {
                cand[j] = p[j] + delta[j];
            }
            clamp(&mut cand, lo, hi, np);
            let new_cost = sse(&cand);
            if new_cost < cost {
                let improve = cost - new_cost;
                p = cand;
                cost = new_cost;
                lambda = (lambda * 0.5).max(1e-12);
                accepted = true;
                if improve < 1e-12 * (1.0 + cost) {
                    accepted = false;
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

    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_tot: f64 = y.iter().map(|&v| (v - mean) * (v - mean)).sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - cost / ss_tot
    } else {
        0.0
    };

    // Undo normalisation: scale-dependent parameters multiply by `scale`.
    // S0 is always p[0]. Offset (if present) is the last param of the model.
    p[0] *= scale;
    (p, r2)
}

fn model_from_str(name: &str) -> PyResult<Model> {
    match name {
        "mono_exp" => Ok(Model::MonoExp),
        "aronen_t1rho" => Ok(Model::AronenT1rho),
        "aronen_t2" => Ok(Model::AronenT2),
        "rausch" => Ok(Model::Rausch),
        "dwi_mono_exp" | "adc" => Ok(Model::DwiMonoExp),
        "dwi_kurtosis" | "kurtosis" => Ok(Model::DwiKurtosis),
        "dwi_ivim" | "ivim" => Ok(Model::DwiIvim),
        "t2star_biexp" | "biexp_t2star" => Ok(Model::T2starBiExp),
        "stretched_exp" | "stretched" => Ok(Model::StretchedExp),
        "expression" | "expr" => Ok(Model::Expression),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown model '{other}'"
        ))),
    }
}

/// Seed defaults per model when a log-linear seed does not apply.
fn make_seed(model: Model, lo: &[f64], hi: &[f64], np: usize, x: &[f64], y: &[f64]) -> [f64; MAX_NP] {
    match model {
        Model::MonoExp
        | Model::AronenT2
        | Model::AronenT1rho
        | Model::Rausch
        | Model::DwiMonoExp => loglinear_seed(x, y, lo, hi, np),
        Model::DwiKurtosis => {
            let mut s = loglinear_seed(x, y, lo, hi, np);
            // p = [S0, D, K]. K seeded near zero.
            s[2] = 0.5 * (lo[2] + hi[2]);
            clamp(&mut s, lo, hi, np);
            s
        }
        Model::DwiIvim => {
            // p = [S0, D, D_star, f]. Initial guess: D ~ tissue, D* ~ 10x larger, f small.
            let mut s = [0.0_f64; MAX_NP];
            s[0] = y.iter().cloned().fold(0.0_f64, f64::max).max(0.5 * (lo[0] + hi[0]));
            s[1] = 0.5 * (lo[1] + hi[1]);
            s[2] = (10.0 * s[1]).min(hi[2]).max(lo[2]);
            s[3] = 0.1_f64.clamp(lo[3], hi[3]);
            clamp(&mut s, lo, hi, np);
            s
        }
        Model::T2starBiExp => {
            // p = [S0, T_short, T_long, f]. Initial guess: T_short small, T_long large.
            let mut s = [0.0_f64; MAX_NP];
            s[0] = y.iter().cloned().fold(0.0_f64, f64::max).max(0.5 * (lo[0] + hi[0]));
            s[1] = (0.5 * (lo[1] + hi[1])).min(20.0);
            s[2] = (0.5 * (lo[2] + hi[2])).max(60.0);
            s[3] = 0.3_f64.clamp(lo[3], hi[3]);
            clamp(&mut s, lo, hi, np);
            s
        }
        Model::StretchedExp => {
            let mut s = loglinear_seed(x, y, lo, hi, np);
            s[2] = 1.0_f64.clamp(lo[2], hi[2]);
            clamp(&mut s, lo, hi, np);
            s
        }
        Model::Expression => {
            let mut s = [0.0_f64; MAX_NP];
            for j in 0..np {
                s[j] = 0.5 * (lo[j] + hi[j]);
            }
            s
        }
    }
}

/// Fit a flat stack of pixel signals. Returns (params (n_pixels, n_params), r2 (n_pixels,)).
#[pyfunction]
#[pyo3(signature = (
    signals, x, lower, upper, model,
    tr=0.0, t1=0.0, alpha=0.0, te=0.0, t2star=0.0,
    normalize=false, max_iter=100, expression=None
))]
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
    expression: Option<&str>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let model_enum = model_from_str(model)?;
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

    // Resolve param count: built-in -> fixed; expression -> from lower shape.
    let np = if matches!(model_enum, Model::Expression) {
        lo.ncols()
    } else {
        model_enum.n_params()
    };
    if np == 0 || np > MAX_NP {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "param count {np} out of range (1..={MAX_NP})"
        )));
    }
    if lo.nrows() != n || hi.nrows() != n || lo.ncols() != np || hi.ncols() != np {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lower/upper must have shape (n_pixels, {np}) for model '{model}'"
        )));
    }

    // Compile expression if requested.
    let compiled_expr = match model_enum {
        Model::Expression => {
            let src = expression.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "model='expression' requires the `expression=` keyword",
                )
            })?;
            Some(expr::compile(src, np).map_err(pyo3::exceptions::PyValueError::new_err)?)
        }
        _ => None,
    };

    let xv: Vec<f64> = xs.iter().cloned().collect();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| sig.row(i).iter().cloned().collect()).collect();
    let los: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..np).map(|j| lo[(i, j)]).collect())
        .collect();
    let his: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..np).map(|j| hi[(i, j)]).collect())
        .collect();

    let results: Vec<([f64; MAX_NP], f64)> = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let seed = make_seed(model_enum, &los[i], &his[i], np, &xv, &rows[i]);
                match &compiled_expr {
                    Some(prog) => {
                        let eval_fn = |xi: f64, p: &[f64]| prog.eval(xi, p);
                        fit_pixel_lm(
                            &eval_fn,
                            &xv,
                            &rows[i],
                            &los[i],
                            &his[i],
                            np,
                            seed,
                            normalize,
                            max_iter,
                        )
                    }
                    None => {
                        let eval_fn = |xi: f64, p: &[f64]| eval_builtin(model_enum, &seq, xi, p);
                        fit_pixel_lm(
                            &eval_fn,
                            &xv,
                            &rows[i],
                            &los[i],
                            &his[i],
                            np,
                            seed,
                            normalize,
                            max_iter,
                        )
                    }
                }
            })
            .collect()
    });

    let mut params = vec![0.0_f64; n * np];
    let mut r2 = vec![0.0_f64; n];
    for (i, (p, r)) in results.iter().enumerate() {
        for j in 0..np {
            params[i * np + j] = p[j];
        }
        r2[i] = *r;
    }

    let params_arr = PyArray1::from_vec_bound(py, params).reshape([n, np])?;
    let r2_arr = r2.into_pyarray_bound(py);
    Ok((params_arr, r2_arr))
}

/// Smoke-check that a user expression compiles and evaluates at a sample point.
/// Returns the value `f(x, p)` so the caller can sanity-check the expression
/// before launching a long volume fit.
#[pyfunction]
fn check_expression(expression: &str, x: f64, params: Vec<f64>) -> PyResult<f64> {
    let prog = expr::compile(expression, params.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(prog.eval(x, &params))
}

#[pymodule]
fn bmri_fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_volume, m)?)?;
    m.add_function(wrap_pyfunction!(check_expression, m)?)?;
    Ok(())
}

// Suppress unused-import warning on builtins-only build paths.
#[allow(dead_code)]
fn _unused_keepalive() {
    let _ = sse_builtin;
}

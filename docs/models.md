# Built-in models

bMRI ships with a curated set of relaxometry and diffusion models. Every
model works with all three backends (`curvefit`, `loglinear`, `rust`) and
supports `region_bounds` for per-label parameter ranges.

## Relaxometry

### Mono-exponential T2 / T2* — `bmri.fitting.T2T2star`

```
S(TE) = S0 * exp(-TE / T) + C
```

Three parameters: `S0`, `T` (T2 or T2* in ms), offset `C`. The Rust model
name is `mono_exp`.

### T1ρ / T2-prep (Aronen) — `bmri.fitting.T1RhoT2prep`

Aronen variants with finite TR, B1+ and T2*-leakage corrections.
Three parameters: `S0`, `T`, offset. Sequence parameters TR, T1, alpha,
TE, T2* are passed as keyword arguments.

### Inversion-Recovery T1 — `bmri.fitting.InversionRecoveryT1`

```
S(TI) = S0 * (1 - 2*exp(-TI/T1)) + C
```

## Diffusion

### ADC mono-exponential — `bmri.fitting.DWIMonoExp`

```
S(b) = S0 * exp(-b * D) + C
```

Three parameters: `S0`, ADC `D` in mm²/s, offset.

### Kurtosis — `bmri.fitting.DWIKurtosis`

```
S(b) = S0 * exp(-b*D + (b*D)² * K / 6)
```

Three parameters: `S0`, `D`, kurtosis `K`. Requires b-values up to about
2000 s/mm² for stable fits.

### IVIM — `bmri.fitting.DWIIvim`

```
S(b) = S0 * (f * exp(-b*D*) + (1-f) * exp(-b*D))
```

Four parameters: `S0`, slow diffusion `D`, pseudo-diffusion `D*`,
perfusion fraction `f`. Needs the Rust backend (scipy can fit it too but
the seeding is tuned for the Rust path). Typical bounds for abdomen:
`D in [0, 5e-3]`, `D* in [5e-3, 1e-1]`, `f in [0, 1]`.

## Multi-component

### Bi-exponential T2* — `bmri.fitting.T2StarBiExp`

```
S(TE) = S0 * (f * exp(-TE/Ts) + (1-f) * exp(-TE/Tl))
```

Four parameters. Useful for myelin water imaging or tissues with one
short and one long T2* component.

### Stretched exponential — `bmri.fitting.StretchedExp`

```
S(x) = S0 * exp(-(x/T)^α)
```

Three parameters: `S0`, `T`, stretch exponent `α`. `α = 1` recovers a
mono-exp.

## Custom expression — `bmri.fitting.CustomExpression`

Define your own model as a string. Both backends use the same syntax.

```python
from bmri.fitting import CustomExpression

model = CustomExpression(
    expression="S0 * exp(-x/T) + C",
    boundary=([0, 0, -1], [10, 200, 1]),
)
fit_maps, r2 = model.fit(data, mask, te, method="rust")
```

Identifiers:

- `x` — independent axis (TE, b-value, TSL, TI)
- `p0`..`p7` — positional parameter access
- `S0`, `T`, `T1`, `T2`, `T2s`, `T1rho`, `D`, `ADC` — aliases for `p1` (the
  characteristic time / coefficient)
- `K`, `D_star`, `T_long`, `Tl` — aliases for `p2`
- `offset`, `C`, `f`, `alpha` — alias for the last parameter (`p3` or `p2`)

Functions: `exp`, `log`, `sin`, `cos`, `tan`, `sqrt`, `abs`, `pow`,
`min`, `max`.

Operators: `+`, `-`, `*`, `/`, `^` (or `**` for power).

### Performance note

The Rust backend compiles the expression to a small stack-based program
once and then evaluates it per voxel in roughly nanoseconds. There is no
Python callback during the fit, so performance matches the built-in
models. The scipy backend uses an AST walker that is about 5-10× slower
per voxel but functionally equivalent.

## Registry

```python
import bmri.fitting as bf
print([cls.__name__ for cls in bf.__all__])
# T2T2star, T1rho_T2prep, InversionRecoveryT1,
# DWIMonoExp, DWIKurtosis, DWIIvim,
# T2StarBiExp, StretchedExp, CustomExpression
```

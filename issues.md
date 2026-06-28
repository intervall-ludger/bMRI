## Pre-existing test failures (unrelated to Rust backend work)

- test/test_AbstraktFitting.py::test_fit[0..3] and test_fit_reshaped_2D_to_3D fail with
  `TypeError: too many arguments`. Reproduces on the unmodified tree (AbstractFitting.py
  and the test were not touched). Likely a numba njit signature mismatch in the fit path.
  Investigate separately.

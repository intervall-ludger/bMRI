# bMRI - Bio-sensitive MRI Analysis Framework

<div align="center">

[![Actions Status](https://github.com/ludgerradke/bMRI/actions/workflows/test.yml/badge.svg)](https://github.com/ludgerradke/bMRI/actions/workflows/test.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Professional CLI tool for MRI relaxation time analysis**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 🎯 Overview

**bMRI** is a modern, production-ready Python framework for analyzing bio-sensitive MRI data. It provides professional command-line tools for fitting relaxation time maps (T1, T2, T2*, T1ρ) from DICOM data with beautiful terminal output, robust error handling, and comprehensive validation.

### What's New in v0.2.0

🚀 **Complete transformation to professional CLI tool:**

- ✨ **Modern CLI** - Beautiful Typer-based interface with Rich terminal output
- 🛡️ **Robust Error Handling** - Structured exceptions and comprehensive validation
- 📝 **Type Safety** - Full type hints with ty support
- ⚙️ **Configuration Management** - TOML-based configs with Pydantic validation
- 📊 **Progress Tracking** - Real-time progress bars and status updates
- 🎨 **Beautiful Output** - Colored, formatted terminal output with Rich
- 🔍 **Professional Logging** - Structured logging with file output support

---

## ✨ Features

### Core Functionality

- **T2/T2* Fitting** - Mono-exponential relaxation time fitting from multi-echo DICOM data
- **T1ρ (T1-rho) Fitting** - Spin-lattice relaxation in rotating frame with multiple models:
  - Mono-exponential decay
  - Aronen model (with sequence parameters)
  - Rausch model
- **T1 Fitting** - Longitudinal relaxation time analysis
- **Multi-ROI Analysis** - Automatic statistics for multiple regions of interest
- **Quality Control** - R² thresholding and pixel-wise validation

### Technical Highlights

- **Parallel Processing** - Multi-core CPU support with automatic detection
- **Numba Acceleration** - JIT-compiled performance-critical code
- **DICOM Support** - Robust DICOM reading with automatic layout detection
- **NIfTI Output** - Standard neuroimaging format for compatibility
- **Flexible Boundaries** - Customizable fitting parameter constraints
- **Progress Reporting** - Real-time feedback during long computations

### CLI Features

- **Type-Safe Arguments** - Automatic validation and type conversion
- **Config Files** - TOML configuration for reproducible analysis
- **Beautiful Help** - Auto-generated, formatted help messages
- **Error Messages** - Clear, actionable error descriptions
- **Logging** - Configurable verbosity and file logging

---

## 📦 Installation

### Prerequisites

**System Dependencies:**

```bash
# macOS
brew install qt

# Ubuntu/Debian
sudo apt-get install qt5-default libgl1-mesa-glx
```

### Using uv (Recommended - 10-100x faster!)

[uv](https://github.com/astral-sh/uv) is a modern, blazingly fast Python package installer.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/ludgerradke/bMRI.git
cd bMRI

# Install bMRI with all dependencies
uv pip install -e ".[dev]"
```

### Using pip (Traditional)

```bash
# Clone repository
git clone https://github.com/ludgerradke/bMRI.git
cd bMRI

# Install in development mode
pip install -e ".[dev]"
```

### Verify Installation

```bash
bmri --version
# Output: bMRI version 0.2.0

bmri --help
# Shows beautiful CLI help
```

---

## 📊 Test Data

Example DICOM datasets are available for testing bMRI. The test data includes:

- **T2* mapping** - 3D multi-echo gradient echo (1056 DICOM files)
- **T2 mapping** - Multi-echo spin echo (300 DICOM files)
- **T1ρ mapping** - Spin-lock prepared sequences (64 DICOM files)
- **Masks** - Pre-segmented NIfTI masks for ROI analysis

**Access:**
- 📦 Test data is available via **sciebo** (secure cloud storage)
- 🔑 To request access credentials, contact: **ludger.radke@med.uni-duesseldorf.de** (University Hospital Düsseldorf)
- 📥 Once downloaded, extract to `test/resources/` in the bMRI directory

The example data is also used by `run_example.py` for quick testing.

---

## 🚀 Quick Start

### Option 1: Run Example Script (Easiest!)

For quick testing with the included example data, use the beautiful Rich-based example runner:

```bash
# Run T2* fitting example
python run_example.py t2star

# Run T2 fitting example
python run_example.py t2

# Run T1ρ fitting example
python run_example.py t1rho

# Run all examples sequentially
python run_example.py all
```

The T2* example automatically launches the interactive viewer so you can inspect the fitted map (close the window to continue).

**Output:**
```
╭─────────────────────────────────────────────╮
│ T2* Relaxation Time Fitting                 │
│ bMRI - Bio-sensitive MRI Analysis Framework │
╰─────────────────────────────────────────────╯

📁 Data:   7_T2-star_map_3D_cor_03445
📊 Files:  1056 DICOM files

⠋ Fitting T2* maps... ━━━━━━━━━━━━━━━━━━━━━━━━ 0:01:23

✓ Fitting Complete!

NIfTI Files:
  → S0_map.nii.gz
  → t2_t2star_map.nii.gz
  → r2.nii.gz
  ...

                    T2* RESULTS
┏━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━━━━━┳━━━━━━┓
┃ ROI ┃  Mean ┃  Std ┃  Min ┃  Max ┃      Pixels ┃   R² ┃
┡━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━━━━━╇━━━━━━┩
│   1 │ 16.78 │ 12.9 │ 4.47 │ 50.0 │ 8255/8534   │ 0.95 │
│   2 │ 20.92 │ 14.3 │ 4.00 │ 50.0 │ 4727/4757   │ 0.98 │
└─────┴───────┴──────┴──────┴──────┴─────────────┴──────┘
```

### Option 2: Command Line Interface

For production use with your own data:

#### Basic T2* Fitting

```bash
# Example with test data (if you have downloaded it)
bmri fit t2star test/resources/Mrtc-Studie_Cartilage_transplantation_05_GAPF97478/20211125_0925/7_T2-star_map_3D_cor_03445 \
    --mask test/resources/Mrtc-Studie_Cartilage_transplantation_05_GAPF97478/20211125_0925/7_T2-star_map_3D_cor_03445/mask.nii.gz \
    --output ./results \
    --min-r2 0.75

# Or with your own data
bmri fit t2star /path/to/dicom/folder \
    --mask /path/to/mask.nii.gz \
    --output ./results \
    --min-r2 0.75
```

#### Visualize T2*/T2 Results

```bash
# Launch the interactive PyQt viewer (expects dicom.nii.gz, params.nii.gz, acquisition_times.txt)
bmri view t2star ./results --parameter t2_t2star
```

Close the viewer window to return to the terminal.

**Output:**
```
╭─────────────────────────────────────────╮
│ T2* Relaxation Time Fitting             │
│ bMRI - Bio-sensitive MRI Analysis       │
╰─────────────────────────────────────────╯

🔍 Validating inputs...
   ✓ DICOM folder: /path/to/dicom (1056 files)
   ✓ Mask: mask.nii.gz (128×128×32, 12450 voxels, 5 ROIs)

Configuration:
  DICOM Folder:  /path/to/dicom
  Mask File:     mask.nii.gz
  Output Dir:    ./results
  Normalize:     True
  Min R²:        0.75
  CPU Cores:     auto

⠋ Fitting T2* maps... ━━━━━━━━━━━━━━━━━━ 0:01:23

✓ Fitting complete!
Results saved to: ./results
```

#### T2 Fitting with Custom Boundaries

```bash
# Example with test data
bmri fit t2 test/resources/Mrtc-Studie_Cartilage_transplantation_05_GAPF97478/20211125_0925/10_T2_map_cor_10282 \
    --mask test/resources/Mrtc-Studie_Cartilage_transplantation_05_GAPF97478/20211125_0925/10_T2_map_cor_10282/mask.nii.gz \
    --boundary "0.9,3;5,40;-0.5,0.5" \
    --normalize \
    --min-r2 0.7 \
    --pools 8

# Or with your own data
bmri fit t2 /path/to/t2/dicom \
    --mask /path/to/mask.nii.gz \
    --boundary "0.9,3;5,40;-0.5,0.5" \
    --normalize \
    --min-r2 0.7 \
    --pools 8
```

**Boundary Format:** `S0_min,S0_max;T2_min,T2_max;offset_min,offset_max`

### T1ρ Fitting with Configuration File

**1. Create config file `t1rho_config.toml`:**

```toml
# Fitting model
model = "mono_exp"  # Options: mono_exp, aronen, rausch

# Parameter boundaries
[[boundary]]
lower = [1, 1, -1000]      # [S0_min, T1rho_min, offset_min]
upper = [10000, 500, 1000]  # [S0_max, T1rho_max, offset_max]

# Fitting options
normalize = false
min_r2 = 0.3
pools = 0  # 0 = auto-detect

# Optional: Sequence parameters for Aronen/Rausch models
[sequence]
TR = 1000.0
T1 = 1500.0
alpha = 90.0
TE = 20.0
T2star = 50.0
```

**2. Run fitting:**

```bash
# Example with test data (assuming T1rho folders are the 4 TSL acquisitions)
bmri fit t1rho test/resources/Mrtc-Studie_Cartilage_transplantation_05_GAPF97478/20211125_0925/12_T1rho_cor_12161 \
    --mask test/resources/Mrtc-Studie_Cartilage_transplantation_05_GAPF97478/20211125_0925/T1rho/mask.nii.gz \
    --tsl "0,10,40,70" \
    --config t1rho_config.toml \
    --output ./t1rho_results

# Or with your own data
bmri fit t1rho /path/to/t1rho/dicom \
    --mask /path/to/mask.nii.gz \
    --tsl "0,10,40,70" \
    --config t1rho_config.toml \
    --output ./t1rho_results
```

---

## 📖 Documentation

### CLI Commands

#### Global Options

```bash
bmri [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose         Enable verbose logging (DEBUG level)
  -q, --quiet           Minimal output (WARNING level only)
  --log-file PATH       Write logs to file
  --version             Show version and exit
  --help                Show help message
```

#### Fit Commands

##### `bmri fit t2star`

Fit T2* relaxation times from multi-echo gradient echo sequences.

```bash
bmri fit t2star DICOM_FOLDER [OPTIONS]

Required Arguments:
  DICOM_FOLDER          Path to folder containing DICOM files

Required Options:
  -m, --mask PATH       NIfTI mask file (.nii or .nii.gz)

Optional:
  -o, --output PATH     Output directory [default: ./bmri_output]
  -c, --config PATH     TOML configuration file
  --boundary TEXT       Custom parameter bounds (see format below)
  --normalize           Normalize signal before fitting [default: true]
  --min-r2 FLOAT        Minimum R² threshold [0.0-1.0] [default: 0.75]
  --pools INTEGER       CPU cores for parallel processing (0=auto) [default: 0]
  --dim INTEGER         Dimensionality: 2 or 3 [default: 3]
```

**Example:**
```bash
bmri fit t2star /data/patient01/t2star_dicom \
    -m /data/patient01/mask.nii.gz \
    --boundary "0.9,2;0,50;-inf,inf" \
    --min-r2 0.8 \
    --pools 8
```

##### `bmri fit t2`

Fit T2 relaxation times from spin echo sequences.

```bash
bmri fit t2 DICOM_FOLDER -m MASK [OPTIONS]

# Same options as t2star, different default boundaries
# Default boundary: "0.9,3;5,40;-0.5,0.5"
# Default min_r2: 0.7
```

##### `bmri fit t1rho`

Fit T1ρ relaxation times with spin-lock preparation.

```bash
bmri fit t1rho DICOM_FOLDER -m MASK --tsl TSL_TIMES [OPTIONS]

Required:
  --tsl TEXT            Spin-lock times (comma-separated)
                        Example: "0,10,40,70"

Additional Options:
  --model TEXT          Fitting model [default: mono_exp]
                        Options: mono_exp, aronen, rausch

# Example
bmri fit t1rho /data/t1rho \
    -m mask.nii.gz \
    --tsl "0,10,40,70" \
    --model mono_exp \
    --boundary "1,10000;1,500;-1000,1000" \
    --min-r2 0.3
```

---

## 🔧 Configuration Files

Configuration files use TOML format for human-readable, validated settings.

### Example: T2* Configuration

```toml
# t2star.toml

# Fitting parameters
normalize = true
min_r2 = 0.75
pools = 8  # Number of CPU cores
dim = 3    # 2D or 3D

# Parameter boundaries: [S0, T2*, offset]
[boundary]
lower = [0.9, 0, "-inf"]
upper = [2.0, 50, "inf"]
```

### Example: T1ρ with Aronen Model

```toml
# t1rho_aronen.toml

model = "aronen"
normalize = false
min_r2 = 0.3
dim = 3

[boundary]
lower = [1, 1, -1000]
upper = [10000, 500, 1000]

# MRI sequence parameters (required for Aronen model)
[sequence]
TR = 1000.0      # Repetition time (ms)
T1 = 1500.0      # T1 relaxation time (ms)
alpha = 90.0     # Flip angle (degrees)
TE = 20.0        # Echo time (ms)
T2star = 50.0    # T2* relaxation time (ms)
```

**Usage:**
```bash
bmri fit t1rho /data/t1rho -m mask.nii.gz --tsl "0,10,40,70" -c t1rho_aronen.toml
```

---

## 💡 Examples

### Example 1: Quick Start with Example Data

The easiest way to get started is using the included example runner with test data:

```bash
# Single example
python run_example.py t2star

# All examples with statistics
python run_example.py all
```

This will:
- ✅ Use pre-configured test data from `test/resources/`
- ✅ Run fitting with optimal parameters
- ✅ Display beautiful progress bars and results
- ✅ Show ROI statistics in formatted tables
- ✅ Generate NIfTI maps and CSV files

**With uv (faster):**
```bash
uv run python run_example.py all
```

### Example 2: Batch Processing Multiple Patients

```bash
#!/bin/bash
# process_patients.sh

PATIENTS=(patient01 patient02 patient03)
DATA_DIR="/data/cartilage_study"

for patient in "${PATIENTS[@]}"; do
    echo "Processing $patient..."

    bmri fit t2star "$DATA_DIR/$patient/t2star" \
        -m "$DATA_DIR/$patient/mask.nii.gz" \
        -o "$DATA_DIR/$patient/results" \
        --min-r2 0.75 \
        --pools 8
done
```

### Example 3: Using Python API (Legacy)

For scripting and integration, you can still use the Python API:

```python
from pathlib import Path
import numpy as np

from src.Fitting import T2_T2star
from src.Utilitis import load_nii

# Setup paths
dicom_folder = Path("/data/patient01/t2star")
mask_file = Path("/data/patient01/mask.nii.gz")

# Initialize fitter
fitter = T2_T2star(
    dim=3,
    boundary=([0.9, 0, -np.inf], [2, 50, np.inf]),
    normalize=True
)

# Run fitting
results = fitter.run(
    dicom_folder=dicom_folder,
    mask_file=mask_file,
    pools=8,
    min_r2=0.75
)

print(f"Fitting completed! Results saved to {dicom_folder}")
```

### Example 4: T1ρ with Manual TSL Generation

```python
from src.Fitting import T1rho_T2prep

# Generate TSL times: first=10ms, increment=30ms, n=4 points
t1rho = T1rho_T2prep(dim=3, config=None, normalize=False)
tsl = t1rho.get_TSL(first_SL=10, inc_SL=30, n=4)
print(tsl)  # [0, 10, 40, 70]

# Run fitting
t1rho.run(
    dicom_folder=Path("/data/t1rho"),
    mask_file=Path("/data/mask.nii.gz"),
    tsl=tsl,
    min_r2=0.3
)
```

---

## 📊 Output Files

After fitting, bMRI generates the following outputs:

### NIfTI Files (`.nii.gz`)

| File | Description |
|------|-------------|
| `S0_map.nii.gz` | Initial signal intensity (S₀) |
| `t2_t2star_map.nii.gz` | T2 or T2* relaxation time map (ms) |
| `offset_map.nii.gz` | Signal offset parameter |
| `r2.nii.gz` | R² goodness-of-fit map |
| `params.nii.gz` | All parameters combined (4D) |
| `mask.nii.gz` | Copy of input mask |
| `dicom.nii.gz` | Original DICOM data in NIfTI format |

### CSV Files

| File | Description |
|------|-------------|
| `{folder}_S0.csv` | ROI statistics for S₀ parameter |
| `{folder}_t2_t2star.csv` | ROI statistics for T2/T2* values |
| `{folder}_offset.csv` | ROI statistics for offset |

**CSV Format:**
```csv
mask_index;mean;std;min;max;Pixels;Mean R^2
1;16.78;12.90;4.47;50.00;8255/8534;0.95
2;20.92;14.32;4.00;50.00;4727/4757;0.98
```

---

## 🏗️ Architecture

### Project Structure

```
bMRI/
├── src/
│   ├── bmri/                      # Modern CLI framework (v0.2.0)
│   │   ├── cli/                   # CLI commands and entry points
│   │   │   ├── main.py            # Main CLI app
│   │   │   └── commands/
│   │   │       └── fit.py         # Fitting commands
│   │   ├── exceptions.py          # Exception hierarchy
│   │   ├── logger.py              # Rich-based logging
│   │   ├── types.py               # Type aliases and protocols
│   │   ├── validators.py          # Input validation
│   │   ├── config.py              # Pydantic configuration
│   │   └── io/
│   │       └── readers.py         # DICOM/NIfTI readers
│   │
│   ├── Fitting/                   # Core fitting algorithms
│   │   ├── AbstractFitting.py     # Base class with parallel processing
│   │   ├── T2_T2star.py           # T2/T2* fitting
│   │   ├── T1rho_T2prep.py        # T1ρ fitting
│   │   ├── T1.py                  # T1 fitting
│   │   └── FittingMap.py          # Post-processing (dGEMRIC, etc.)
│   │
│   ├── Utilitis/                  # I/O utilities
│   │   ├── read.py                # Legacy DICOM readers
│   │   ├── results_writer.py      # Save results as CSV/NIfTI
│   │   └── utils.py               # Helper functions
│   │
│   └── Visualization/             # PyQt5 GUI viewer
│       └── image_viewer.py        # Interactive DICOM/map viewer
│
├── test/                          # Pytest test suite
│   ├── test_T2.py
│   ├── test_T1.py
│   └── ...
│
├── pyproject.toml                 # Project metadata and dependencies
├── README.md                      # This file
└── run_example.py                 # Legacy example runner
```

### Design Patterns

- **Template Method Pattern** - `AbstractFitting` defines fitting workflow
- **Strategy Pattern** - Multiple fitting models (mono-exp, Aronen, Rausch)
- **Factory Pattern** - Configuration-based model selection
- **Dataclass Pattern** - Structured data with `NIfTIMask`, configs

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest test/test_T2.py -v

# Run with verbose output
pytest -vv
```

### Code Quality

```bash
# Format code with black
black src/

# Type checking with ty (from Astral)
uv run ty check src/bmri

# Run pre-commit hooks
pre-commit run --all-files
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/ludgerradke/bMRI.git
cd bMRI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
bmri --version
pytest
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs** - Open an issue with detailed description and reproduction steps
2. **Suggest Features** - Describe the feature and use case
3. **Submit Pull Requests** - Follow the code style and include tests

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the code style:
   - Use type hints for all functions
   - Write Google-style docstrings in English
   - Add tests for new functionality
   - Run `black` and `pytest` before committing
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Style

- **Python 3.10+** - Use modern Python features
- **Type Hints** - All functions must have type annotations
- **Docstrings** - Google style in English
- **Line Length** - Max 100 characters (black default)
- **Imports** - Sorted with isort
- **Testing** - Maintain >80% code coverage

---

## 📝 Citation

If you use bMRI in your research, please cite:

```bibtex
@software{bmri2024,
  title = {bMRI: Bio-sensitive MRI Analysis Framework},
  author = {Radke, Ludger},
  year = {2024},
  url = {https://github.com/ludgerradke/bMRI},
  version = {0.2.0}
}
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NumPy** and **SciPy** - Core scientific computing
- **Numba** - JIT compilation for performance
- **PyDICOM** - DICOM file handling
- **NiBabel** - NIfTI format support
- **Typer** - Modern CLI framework
- **Rich** - Beautiful terminal formatting
- **Pydantic** - Data validation

---

## 📞 Contact & Support

- **GitHub Issues**: [github.com/ludgerradke/bMRI/issues](https://github.com/ludgerradke/bMRI/issues)
- **Email**: ludger.radke@med.uni-duesseldorf.de

---

<div align="center">

[⬆ Back to Top](#bmri---bio-sensitive-mri-analysis-framework)

</div>

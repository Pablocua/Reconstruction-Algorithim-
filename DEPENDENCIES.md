# Reconstruction-Algorithm Project: Dependency Management Guide

## 📋 Project Overview
This project implements a neural network-based reconstruction algorithm for predicting the primordial power spectrum (Pk) from Planck CMB observations.

**Author:** Pablo Cuadrado (University of Sussex)  
**Email:** pcuadradolo97@gmail.com

---

## 🎯 Core Dependencies Explained

### Scientific Computing
| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥1.24.0 | Numerical arrays, mathematical operations for Pk/Cl calculations |
| **scipy** | ≥1.8.0 | Scientific functions (FortranFile for covariance matrices) |

### Machine Learning
| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | ≥2.0.0 | PyTorch neural network framework (Sequential, ReLU, optimizers) |

### Cosmology
| Package | Version | Purpose |
|---------|---------|---------|
| **camb** | ≥1.3.0 | Cosmological simulations & CMB power spectrum calculation |

### Visualization
| Package | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | ≥3.5.0 | 2D plotting (gridspec, error bars, log-log plots) |

### Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| **tqdm** | ≥4.64.0 | Progress bars for training loops |

### Development (Optional)
| Package | Version | Purpose |
|---------|---------|---------|
| **pytest** | ≥7.0.0 | Unit testing framework |
| **pytest-cov** | ≥4.0.0 | Code coverage reporting |
| **jupyter** | ≥1.0.0 | Interactive notebooks |
| **ipython** | ≥8.0.0 | Enhanced Python shell |

---

## 🚀 Installation Instructions

### Step 1: Activate Virtual Environment
```bash
source /Users/pablo/Desktop/Pablo/worksapce/.venv/bin/activate
# Verify: prompt should show (.venv)
```

### Step 2: Install Project Dependencies
```bash
# Navigate to project directory
cd /Users/pablo/Desktop/Pablo/worksapce/Reconstruction-Algorithim-

# Install requirements
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
# Check all packages installed
pip list | grep -E "torch|camb|numpy|scipy|matplotlib|tqdm"

# Quick test imports
python -c "import torch; import camb; import numpy; print('✅ All dependencies installed!')"
```

### ⚠️ Special Notes for This Project

#### **CAMB Installation**
CAMB requires compilation and may need:
- **macOS**: Xcode command line tools
  ```bash
  xcode-select --install
  ```
- **Linux**: gcc compiler
  ```bash
  sudo apt-get install build-essential
  ```
- **Windows**: Microsoft C++ Build Tools

#### **PyTorch GPU Support (Optional)**
If you want GPU acceleration with your GPU:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only (already installed)
pip install torch
```

The code automatically detects GPU: `device = 'cuda:0' if torch.cuda.is_available() else 'cpu'`

---

## 📊 Project Structure

```
Reconstruction-Algorithim-/
├── Neural_N.py                    # Main module (1176 lines)
├── requirements.txt               # Dependency list (this file)
├── DEPENDENCIES.md               # Documentation (this file)
├── PlanckData.txt                # Planck observational data
├── PlanckBinned.txt              # Binned Planck data
├── cls_binned_cov.npy            # Covariance matrix
└── data/
    ├── plik_lite_v22.dataset     # CMB power spectrum data
    ├── cl_cmb_plik_v22.dat       # Binned Cl values
    ├── blmin.dat                 # Bin minimum multipole
    ├── blmax.dat                 # Bin maximum multipole
    ├── bweight.dat               # Bin weights
    └── c_matrix_plik_v22.dat     # Covariance matrix (Fortran binary)
```

---

## 🔧 Architecture Overview

### Key Components

#### **1. Data Generation** (`DataGenerator`, `dataset` functions)
- Generates synthetic training datasets with different Pk models
- Options: Power-law, Fourier series, polynomial perturbations
- Normalizes Cl and Pk for neural network input
- Splits into 80% train, 20% validation

#### **2. Neural Network** (`Plot_training`, `Final_Prediction`)
```
Input (215 Cl values) 
  ↓
Linear(215 → 300) + ReLU
  ↓
Linear(300 → 300) + ReLU [×4 layers]
  ↓
Linear(300 → 215) → Output Pk values
```

#### **3. Cosmology Computations** (CAMB-based)
- `get_results()` - Initialize CAMB with fiducial cosmology
- `update_results()` - Update Pk in CAMB results
- `get_cmb_cls()` - Extract temperature (TT) Cl from results
- `CMBNativeSimulator` - Binning and noise injection

#### **4. Perturbations**
- **Power-law** Pk: Simple power law
- **Fourier** Pk: Power law + Fourier series perturbation
- **Polynomial** Pk: Power law + 5th degree polynomial perturbation

---

## 💾 Input/Output Files

### Data Files (Required)
These files must be in the project directory:
- **PlanckData.txt** - Raw Planck measurements (ℓ, Dℓ, error_lower, error_upper)
- **PlanckBinned.txt** - Binned Planck data
- **data/plik_lite_v22.dataset** - CMB dataset
- **data/cl_cmb_plik_v22.dat** - Pre-computed Cl values
- **data/blmin.dat, blmax.dat, bweight.dat** - Binning information
- **data/c_matrix_plik_v22.dat** - Covariance matrix (Fortran binary format)

### Generated Output Files
When running:
```
Power_law_train.png           # Training results for power-law model
Fourier_train.png             # Training results for Fourier model
Polynomial_train.png          # Training results for polynomial model
Power_law_predss.png          # Predictions from Planck data
Fourier_predss.png
Polynomial_predss.png
Predictions.png               # Comparison of 3 models
epochs.png                    # Epoch comparison plots
{5000,10000,15000,20000}.txt  # Prediction outputs for different epochs
```

---

## 🛠️ Usage Examples

### Example 1: Train Power-Law Model
```python
from Neural_N import Plot_training

# Train for 1000 epochs
Plot_training(mode='Power_law', n_epochs=1000)
# Generates: Power_law_train.png, Power_law_predss.png
```

### Example 2: Train All Models & Compare
```python
from Neural_N import Plot_training

for mode in ['Power_law', 'Fourier', 'Polynomial']:
    Plot_training(mode=mode, n_epochs=5000)
```

### Example 3: Interactive Plotting
```python
from Neural_N import Plots, Planclk_Plot

# Plot Pk and Cl for Fourier model
Plots(mode='Fourier')

# Display Planck data
Planclk_Plot()
```

### Example 4: Make Predictions with Noise
```python
from Neural_N import Final_Prediction

# Generate 10 noisy predictions
Final_Prediction(mode='Fourier', n_epochs=5000)
```

---

## 🔍 Dependency Dependency Tree

```
Neural_N.py
├── camb (cosmology simulations)
│   └── scipy (matrix operations)
├── torch (neural networks)
│   └── numpy (arrays)
├── numpy (numerical computing)
├── scipy (FortranFile for covariance)
├── matplotlib (visualization)
└── tqdm (progress bars)
```

---

## 📈 Troubleshooting

### Issue: CAMB Installation Fails
**Solution:**
```bash
# Install system dependencies first
# macOS:
xcode-select --install

# Then reinstall CAMB
pip install --no-cache-dir camb
```

### Issue: "ModuleNotFoundError: No module named 'camb'"
**Solution:**
```bash
# Verify venv is activated
which python  # Should show .venv path

# Reinstall in current environment
pip install camb
```

### Issue: Covariance Matrix File Not Found
**Solution:**
The code expects `data/c_matrix_plik_v22.dat`. Ensure:
1. File exists in `data/` subdirectory
2. Relative path is correct (code runs from project root)
3. File is not corrupted (binary Fortran format)

### Issue: GPU Not Detected
**Solution:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Shows GPU name

# If False, install GPU-specific PyTorch:
# See "PyTorch GPU Support" section above
```

---

## 🧪 Running Tests (Optional)

If test files exist:
```bash
# Run all tests with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_neural_network.py -v

# Run with output
pytest -s
```

---

## 📦 Dependency Version Compatibility

| Python | torch | camb | numpy |
|--------|-------|------|-------|
| 3.9    | ≥2.0  | ≥1.3 | ≥1.24 |
| 3.10   | ≥2.0  | ≥1.3 | ≥1.24 |
| 3.11   | ≥2.0  | ≥1.3 | ≥1.24 |
| 3.12   | ≥2.1  | ≥1.3 | ≥1.24 |

**Current Environment:** Python 3.11.8 ✅

---

## 📚 Additional Resources

- **CAMB Documentation:** https://camb.readthedocs.io/
- **PyTorch Documentation:** https://pytorch.org/docs/
- **Planck Mission Data:** https://pla.esac.esa.int/pla/#home
- **NumPy Guide:** https://numpy.org/doc/
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/

---

## ✅ Installation Checklist

- [ ] Virtual environment activated
- [ ] Python 3.11.8 verified
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Test imports successful
- [ ] Data files present in correct locations
- [ ] GPU support configured (if needed)
- [ ] Ready to run main scripts

---

**Last Updated:** March 24, 2026  
**Status:** Ready for Development ✅

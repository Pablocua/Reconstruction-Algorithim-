# Quick Reference Guide

## 🚀 Quick Start (Copy & Paste)

### 1. Activate Environment
```bash
source /Users/pablo/Desktop/Pablo/worksapce/.venv/bin/activate
```

### 2. Navigate to Project
```bash
cd /Users/pablo/Desktop/Pablo/worksapce/Reconstruction-Algorithim-
```

### 3. Verify Installation
```bash
python -c "import torch, camb, numpy; print('✅ Ready to go!')"
```

---

## 📖 Main Usage Patterns

### Option A: Train Single Model
```python
from Neural_N import Plot_training

# Train Power-Law model
Plot_training(mode='Power_law', n_epochs=1000)

# Outputs: Power_law_train.png, Power_law_predss.png
```

### Option B: Train All Models
```python
from Neural_N import Plot_training

for mode in ['Power_law', 'Fourier', 'Polynomial']:
    Plot_training(mode=mode, n_epochs=5000)
```

### Option C: Make Predictions with Uncertainty
```python
from Neural_N import Final_Prediction

# Generate 10 noisy predictions
Final_Prediction(mode='Fourier', n_epochs=5000)
```

### Option D: Interactive Jupyter
```bash
jupyter notebook
# Then run: from Neural_N import * ; Plot_training('Power_law', 1000)
```

---

## 📊 Available Functions

| Function | Purpose | Usage |
|----------|---------|-------|
| `Plot_training()` | Train & plot results | `Plot_training('Power_law', 5000)` |
| `Final_Prediction()` | Predict with uncertainty | `Final_Prediction('Fourier', 5000)` |
| `Plots()` | Plot Pk and Cl | `Plots('Polynomial')` |
| `Planclk_Plot()` | Show Planck data | `Planclk_Plot()` |
| `Compare_preds()` | Compare all 3 models | `Compare_preds()` |
| `Noisy()` | Show noise effects | `Noisy()` |

---

## 🖥️ Mode Options

```
'Power_law'    → Original power-law expression
'Fourier'      → Power-law + Fourier perturbation
'Polynomial'   → Power-law + polynomial perturbation
'All'          → Randomly mix all three
```

---

## 📈 Typical Workflow

```
1. Activate venv
   ↓
2. Import module
   ↓
3. Call Plot_training() with mode + epochs
   ↓
4. Training runs (shows progress bar)
   ↓
5. Results saved as PNG
   ↓
6. Neural network trained & ready
```

---

## ⏱️ Performance Notes

- **Training Time:** ~1 min per 1000 epochs
- **Data Generation:** Automatic within training
- **Plotting:** Included in training function
- **GPU:** Automatic if available (see DEPENDENCIES.md)

---

## 📁 Generated Files

After running `Plot_training('Power_law', 1000)`:
```
Power_law_train.png       ← Network performance
Power_law_predss.png      ← Predictions
5000.txt                  ← Numeric results
```

---

## 🔗 Full Documentation

- **Installation:** `INSTALLATION_REPORT.md`
- **Dependencies:** `DEPENDENCIES.md`
- **Code:** `Neural_N.py`

---

## ❓ Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Check packages
pip list | grep -E "torch|camb|numpy"

# Install missing package
pip install torch

# Update package
pip install --upgrade torch

# Verify Python
python --version

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Run tests
pytest --cov=.

# Start Jupyter
jupyter notebook

# Deactivate venv
deactivate
```

---

## 💡 Tips

1. **Always activate venv first:** `source .venv/bin/activate`
2. **Run from project directory:** `cd Reconstruction-Algorithim-`
3. **Training is automatic:** Don't need to generate data yourself
4. **Outputs are PNG:** Check them to verify training worked
5. **GPU optional:** CPU is fine, but slower

---

**Ready? Run this in terminal:**
```bash
source /Users/pablo/Desktop/Pablo/worksapce/.venv/bin/activate && \
cd /Users/pablo/Desktop/Pablo/worksapce/Reconstruction-Algorithim- && \
python -c "from Neural_N import Plot_training; Plot_training('Power_law', 1000)"
```

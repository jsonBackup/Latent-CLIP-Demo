
# Supplementary Directory

This repository contains resources and code for utilizing **Latent CLIP** for zero-shot prediction and reward-based noise optimization.

---

## 📋 Installation

To set up the environment, use the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate latentclipenv
```

---

## 🚀 Usage

The starting point for understanding and utilizing **Latent CLIP** is the Jupyter Notebook:

**`minimal_usage_latent_clip.ipynb`**

This notebook demonstrates:
- **Zero-shot prediction** using Latent CLIP.
- **Reward-based noise optimization** using Latent CLIP-based rewards .

---

### 📌 Using the ComfyUI Workflow
The file **`workflow_sdxl_turbo.json`** is a workflow designed for use with **ComfyUI**.

**Steps to use the workflow:**
1. **Clone ComfyUI** from GitHub:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   ```

2. **Run ComfyUI**:
   ```bash
   python main.py
   ```

3. **Load the Workflow**:
   - In the ComfyUI interface, press **`Ctrl + O`** (or click "Load").
   - Select the file **`workflow_sdxl_turbo.json`** from the `assets/` folder.

4. For more information on ComfyUI, visit:  
   ➡️ [ComfyUI GitHub Repository](https://github.com/comfyanonymous/ComfyUI)

---

## 📂 Directory Structure

```
supplementary/
│
├── assets/            # Additional resources
│   └── workflow_sdxl_turbo.json  # ComfyUI workflow file (https://github.com/comfyanonymous/ComfyUI)
│
├── Latent_ReNO/       # Implementation for reward-based noise optimization
├── environment.yml    # Conda environment setup file
├── helper.py          # Utility functions for supporting the notebook
└── minimal_usage_latent_clip.ipynb  # Main notebook for starting with Latent CLIP
```

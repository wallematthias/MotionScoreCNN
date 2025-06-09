# MotionScoreCNN

<img width="452" alt="image" src="https://github.com/OpenMSKImaging/MotionScoreCNN/assets/92020703/f4d8da86-4769-46b0-8eb5-dbd91b379762">

Motion scoring for HR-pQCT using deep convolutional neural networks  
Developed by Matthias Walle and collaborators  
Citation: Walle et al., Bone (2023). https://doi.org/10.1016/j.bone.2022.116607

---

## Overview

`motionscore` is a Python 3.8 command-line tool for automated and manual motion grading of high-resolution quantitative computed tomography (HR-pQCT) scans. It uses an ensemble of deep neural networks to assess motion artifacts in 3D image volumes and generate visual diagnostics.

---

## Requirements

This tool is **limited to Python 3.8** due to the Scanco AIM file format loader.

---

## Installation

We recommend installing in a dedicated conda environment:

```bash
# Create and activate environment
conda create -n motionscore python=3.8 -y
conda activate motionscore

# Install from GitHub
pip install git+https://github.com/wallematthias/MotionScoreCNN

# macOS users
pip install .[mac]

# Linux/Windows users
pip install .[unix]
```

---

## Usage

The CLI supports two modes: `grade` and `confirm`.

### Grade mode — automatic motion scoring

```bash
motionscore grade \
  --input path/to/*.AIM \
  --stackheight 168 \
  --output path/to/output/
```

- Loads and scores each AIM image
- Saves PNG visualizations
- Prints stack and mean motion scores

### Confirm mode — manual review of PNGs

```bash
motionscore confirm \
  --input path/to/*motion.png \
  --threshold 75 \
  --output grades.csv
```

- Automatically accepts high-confidence predictions (e.g. over threshold 75%)
- Displays low-confidence cases for manual review (enter new score in command line / empty = accept default)
- Saves output and accuracy in a CSV

---

## Citation

If you use this software, please cite:

Walle, M., Eggemann, D., Atkins, P.R., Kendall, J.J., Stock, K., Müller, R. and Collins, C.J., 2023.  
Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks.  
*Bone*, 166, p.116607.  
https://doi.org/10.1016/j.bone.2022.116607

---

## License

This project is licensed under the MIT License.

---


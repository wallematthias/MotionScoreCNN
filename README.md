# MotionScoreCNN

<img width="452" alt="image" src="https://github.com/OpenMSKImaging/MotionScoreCNN/assets/92020703/f4d8da86-4769-46b0-8eb5-dbd91b379762">

Motion scoring for HR-pQCT using deep convolutional neural networks  
Re-implementation 2025 by Matthias Walle and collaborators  
Original Citation: Walle et al., Bone (2023). https://doi.org/10.1016/j.bone.2022.116607

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

# Clone the repository
git clone https://github.com/wallematthias/MotionScoreCNN.git
cd MotionScoreCNN

# macOS users
pip install -e .[mac]

# Linux/Windows users
pip install -e .[unix]
```

- Note: Tested on Apple Silicon (M3) install tensorflow manually if running in issues with freezing.
---

### Download Model Weights

Before running the tool, download the pretrained model weights:

1. Visit the following form to request download access:  
   [Request Link](https://forms.gle/cy6wkX83pgKvP5Z69)

2. Fill out the short request form. Access is typically granted within 5 minutes.

3. Download the `.h5` model files named `DNN_0.h5` to `DNN_9.h5` and place them in the `motionscore/models/` folder inside the repository:

```bash
MotionScoreCNN/
├── motionscore/
│   ├── models/
│   │   ├── DNN_0.h5
│   │   ├── DNN_1.h5
│   │   ├── ...
│   │   └── DNN_9.h5
```

The tool will automatically load these models for prediction.

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
- Note: When running this for the first time initialisation can be slow due to tensorflow. 

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

## Workflow

The typical workflow involves two steps:

1. **Automatic Grading**  
   Use `grade` mode to analyze a set of `.AIM` scans and generate motion scores. This will output PNG images visualizing slice-wise predictions.

2. **Manual Confirmation and Aggregation**  
   Use `confirm` mode on the saved PNGs to review automatic scores. High-confidence scores are accepted automatically, while low-confidence scores are shown for manual input. Final results are saved in a `.csv` file with accuracy metrics.

This process ensures robust motion grading while enabling human oversight where needed.

---

## Citation

If you use this software, please cite:

Walle, M., Eggemann, D., Atkins, P.R., Kendall, J.J., Stock, K., Müller, R. and Collins, C.J., 2023. Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks. *Bone*, 166, p.116607. https://doi.org/10.1016/j.bone.2022.116607

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.

---


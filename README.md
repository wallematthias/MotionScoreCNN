# README for Stack Shift Artefact Correction Algorithm

This repository contains the implementation of a method for automatic HR-pQCT motion grading. 

## Installation

Requires: aim package
pip install git+https://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@github.com/OpenMSKImaging/lightAIMwrapper/

1. Open your terminal or command prompt.

2. Run the following command to install the package via pip:
pip install git+https://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@github.com/OpenMSKImaging/MotionScoreCNN/

3. If the package has any dependencies, pip will automatically download and install them for you.

4. Once the installation is complete, you will be able to use `motionscore` as command line funciton. With the following command line arguments. When running the script from command line, it will ask for the input path to the aim files. Provide a glob command (e.g. *.AIM to grade all AIM files in the current directory). Provide a stackheight (usually 168 for standard protocol) and a output path for the individual grades to be saved to. 

5. Use the visual_inspection.ipynb to correct and summarise all grades in a single *.csv file. Right now manual correction only works for single stack I believe. 

6. If you need to upgrade or uninstall the package at a later time, you can use the following commands:
pip install --upgrade git+https://github.com/username/repo-name.git
pip uninstall package-name


Example Use:

<img width="452" alt="image" src="https://github.com/OpenMSKImaging/MotionScoreCNN/assets/92020703/f4d8da86-4769-46b0-8eb5-dbd91b379762">

Please Cite:

https://doi.org/10.1016/j.bone.2022.116607 

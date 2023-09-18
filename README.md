# FHR Signal Generation using CTGGAN

## Introduction

This project revolves around the generation of Fetal Heart Rate (FHR) signals using a 1D Convolutional Neural Network (CNN) trained with a Wasserstein Generative Adversarial Network (WGAN). The objective of this project is to facilitate the synthetic generation of FHR signals which can potentially aid in a variety of clinical and research settings. The dataset used for this project can be downloaded from the following open-source repository: [CTU-UHB CTG Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/).

## Files

- `CTGGAN.py`: The main source code file containing the implementation of the GAN model for FHR signal generation.

## Visualization

### Generated Signal Samples

![Generated Signal Samples](myplot.png)

### Loss Function for Model Training

![Loss Function](loss.png)

## Setup 

To set up and install the necessary environments and dependencies, follow the steps below:

1. Ensure that you have Python 3.9 and pytorch installed on your system. You can download it from [here](https://www.python.org/downloads/).
3. Clone the repository to your local machine.
3. Navigate to the project directory in your terminal.
4. Install the necessary packages mentioned in - `CTGGAN.py`

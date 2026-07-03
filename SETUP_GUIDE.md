# DanceSport Detection Framework – Setup Guide

**Author:** Fatema Hamidi

This guide explains how to install, configure, and run the DanceSport Detection Framework on a new computer.

---

# What you will install

| Software | Purpose |
|-----------|---------|
| Python 3.11 | Runs the framework |
| Git | Downloads the project |
| Visual Studio Code (optional) | Code editor |
| Python Packages | Required libraries |
| Streamlit | Graphical User Interface |
| Ultralytics YOLO | Object Detection Framework |

Minimum requirements

- Python 3.11
- Approximately 5 GB free disk space
- 8 GB RAM

Recommended

- 16 GB RAM
- NVIDIA GPU with CUDA support (optional)

The framework was developed and tested on a **2017 MacBook Air** equipped with an Intel Core i5 processor and integrated Intel HD Graphics without CUDA acceleration.

If a compatible NVIDIA GPU is available, the training script automatically detects it and performs model training using CUDA. Otherwise, the framework automatically falls back to CPU execution.

---

# Part 1 — Install Python

Go to

https://www.python.org/downloads/

Download the latest **Python 3.11** version.

During installation make sure that you enable

```
Add Python to PATH
```

After installation verify that Python works.

Open Terminal (macOS)

or

Command Prompt (Windows)

Run

```bash
python --version
```

or

```bash
python3 --version
```

Example

```
Python 3.11.10
```

---

# Part 2 — Install Git

Download Git

https://git-scm.com/downloads

Install using the default settings.

Verify the installation

```bash
git --version
```

Example

```
git version 2.xx.x
```

---

# Part 3 — Download the Project

There are two ways to obtain the Dancesport Detection Framework.

## Option 1 — Download from GitHub

Clone the GitHub repository:

```bash
git clone https://github.com/fa217/dancesport-detection-framework.git
```

Navigate to the project directory:

```bash
cd dancesport-detection-framework
```

---

## Option 2 — Use the ZIP Archive

If you received the project as a ZIP archive:

1. Download the ZIP file.
2. Extract the archive to a location of your choice.
3. Open the extracted project folder.

# Before You Start

This GitHub repository contains the complete DanceSport Detection Framework, including:

- Source code
- Streamlit application
- Trained YOLO models (`runs/detect/`)
- Evaluation results (`runs/evaluation/`)
- Test results (`runs/test/`)
- Video detection results (`runs/videos/`)
- Formation analysis results

Due to GitHub's file size limitations, the following resources are **not included** in this repository:

- Datasets
- all_formation_images
- Example videos

These files are only required if you want to:

- train the YOLO models,
- reproduce the training and evaluation process,
- evaluate the models using the original datasets,
- browse the datasets in the Streamlit application,
- or perform detection on videos.

The datasets and example videos can be downloaded from the following u:cloud link:

**<https://ucloud.univie.ac.at/index.php/s/3i4CSercWDFkges>**

After downloading the files from u:cloud, copy the folders into the project directory so that the following structure is available:

```text
all_formation_images/

datasets/

videos/
```

The framework is ready to use after installing the required Python packages. No additional model configuration is required, as the trained models and all generated results are already included in the repository.

# Part 4 — Create a Virtual Environment

Open a terminal and navigate to the project directory.

Create a Python virtual environment.

**macOS**

```bash
python3 -m venv .venv
```

**Windows**

```bash
python -m venv .venv
```

This creates a virtual environment named `.venv` inside the project directory.

---

# Part 5 — Activate the Virtual Environment

Activate the virtual environment before installing the required packages.

**macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

If the activation was successful, the terminal prompt should begin with

```text
(.venv)
```

---

# Part 6 — Install the Required Packages

Install all required Python packages by running

```bash
pip install -r requirements.txt
```

The installation may take several minutes.

The framework automatically installs all required dependencies, including

- ultralytics
- torch
- streamlit
- opencv-python
- numpy
- pandas
- pillow
- matplotlib

---

# Part 7 — Verify the Installation

Verify that the installation was successful.

Start the Python interpreter:

```bash
python
```

Then execute:

```python
import ultralytics
import streamlit
import cv2
import torch
```

If no error messages appear, all required packages have been installed successfully.

To exit the Python interpreter, type

```python
exit()
```

---

# Part 8 — Project Structure

After installation the project should look similar to

```text
bachelor_dancesport/
├── all_formation_image/ 
├── datasets/
├── formation_error_analysis/
├── formation_visualizations/
├── framework/
├── runs/
├── src/
├── videos/
├── app.py/
├── CSV_files/
├── requirements.txt/
├── README.md
├── SETUP_GUIDE.md
└── yolo26n.pt
```

---

# Part 9 — Trained YOLO Models

The Streamlit application automatically loads the trained YOLO models from the locations defined in `app.py`.

The default model paths are

```text
runs/detect/dancer_category_model/weights/best.pt

runs/detect/dance_style_model/weights/best.pt

runs/detect/formation_couple_model/weights/best.pt
```

If you downloaded the trained models from the provided u:cloud link, simply copy the complete `runs` folder into the project directory.

No additional configuration is required.

---

# Part 10 — Start the Streamlit Application

Open the project folder.

Run

```bash
streamlit run app.py
```

The browser should automatically open

```
http://localhost:8501
```

If the browser does not open automatically, copy the address into your browser manually.

The application automatically loads

- the selected YOLO model,
- the corresponding dataset,
- and the associated evaluation results

based on the selected model.

No manual configuration is necessary.

---

# Part 11 — Detection Page

The Detection page allows users to perform object detection on images and videos.

## Available models

- Dancer Category
- Dance Style
- Formation Couple

Users can

- upload images
- upload videos
- browse the dataset
- adjust the confidence threshold
- filter detected classes
- download processed videos

---

## Upload Image

1. Select a YOLO model.

2. Click

```
Upload Image
```

3. Select one or more images.

4. The framework automatically performs detection.

The detected objects are displayed together with

- bounding boxes

- class labels

- confidence scores

A detection summary table is generated below the image.

---

## Upload Video

1. Select

```
Upload Video
```

2. Choose a video file.

3. Click

```
Run Video Detection
```

The framework processes the complete video frame by frame.

After processing

- the detected video is displayed

- the processed video can be downloaded

The output video is automatically stored inside

```
framework/output/videos/
```

---

## Browse Dataset

Instead of uploading your own images you may also browse the included datasets.

Available splits

- train

- validation

- test

Simply choose an image and the framework performs detection automatically.

---

# Part 12 — Evaluation Page

The Evaluation page displays the results obtained during YOLO training.

Available information includes

- Precision

- Recall

- mAP50

- mAP50-95

- Confusion Matrix

- Normalized Confusion Matrix

- Precision–Recall Curve

- F1 Curve

- Validation Predictions

The results are loaded automatically from the training folders.

---

# Part 13 — Formation Analysis Page

The Formation Analysis page performs spatial analysis based on the detected dance couples.

The framework calculates

- number of detected couples

- nearest-neighbor distances

- average spacing

- standard deviation

- coefficient of variation

These values are used to evaluate the regularity of DanceSport formations.

The page also displays

- statistical charts

- CSV tables

- formation visualizations

---

# Part 14 — Training the Models

Model training is performed outside the Streamlit application.

To start training, run

```bash
python src/train_yolo.py
```

Before starting the training, open `src/train_yolo.py` and select the dataset by setting

```python
SELECTED_DATASET = "dance_style"

# SELECTED_DATASET = "dancer_category"
# SELECTED_DATASET = "formation_couple"
```

Only one dataset should be selected at a time.

The training configuration, including the number of epochs, image size, and batch size, is defined in the `DATASETS` dictionary inside `train_yolo.py`.

The training script automatically detects whether a compatible NVIDIA GPU with CUDA support is available.

- If a GPU is available, training is performed on the GPU.
- Otherwise, the framework automatically falls back to CPU training.

After training has finished, the trained model weights and evaluation results are stored in

```text
runs/detect/
```

# Support

If any problems occur during installation or execution, please verify that

- Python 3.11 is installed,
- the virtual environment is activated,
- all packages from `requirements.txt` are installed,
- the required datasets, trained models, and videos have been downloaded from u:cloud,
- the folder structure matches the project structure described in this guide.

Following these steps should allow the complete Dancesport Detection Framework to be reproduced successfully.
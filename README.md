# DanceSport Detection Framework

This repository contains the implementation of the Bachelor thesis project **DanceSport Detection Framework**.

The project uses YOLO-based object detection models to analyze DanceSport images and videos. The main focus of the project is the detection of individual dance couples inside formation dance scenes and the extraction of their spatial coordinates for further formation analysis.

## 1. Project Overview

The framework consists of three trained YOLO models:

1. **Dancer Category Model**

   * Detects dancer-related categories.
   * Classes:

     * `dance_couple`
     * `female_dancer`
     * `formation`
     * `male_dancer`

2. **Dance Style Model**

   * Detects the dance style.
   * Classes:

     * `latin`
     * `standard`

3. **Formation Couple Model**

   * Detects individual couples inside formation dance scenes.
   * Class:

     * `couple`

The **Formation Couple Model** is the central model of this project. Its detections are used to extract couple coordinates and perform nearest-neighbor based formation analysis.

---

## 2. Requirements

The project was developed and tested using the following software:

| Component | Version / Specification |
|----------|--------------------------|
| Python | 3.11 |
| Ultralytics | YOLO26n |
| PyTorch | Latest compatible version |
| OpenCV | Latest compatible version |
| NumPy | Latest compatible version |
| Pandas | Latest compatible version |
| Matplotlib | Latest compatible version |
| Pillow | Latest compatible version |
| Streamlit | Latest compatible version |

The experiments were performed on a CPU-based system:

| Component | Specification |
|----------|---------------|
| Device | MacBook Air (2017) |
| Processor | Intel Core i5 |
| Memory | 8 GB RAM |
| Graphics | Intel HD Graphics 6000 |
| GPU | No CUDA-capable GPU |

---

## 3. Installation

### 3.1 Create a Virtual Environment

From the project root directory, create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

---

### 3.2 Install the Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies manually:

```bash
pip install ultralytics streamlit opencv-python pandas numpy matplotlib pillow
```

To verify that Ultralytics is installed correctly, run:

```bash
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
```

## 4. Project Structure

The project uses the following structure:

# Project Structure

```text
bachelor_dancesport/
│
├── datasets/
│   ├── dancer_category/
│   ├── dance_style/
│   └── formation_couple/
│
├── formation_error_analysis/
├── formation_visualization/
├── framework/
│
├── runs/
│   ├── detect/
│   ├── evaluation/
│   ├── test/
│   └── videos/
│
├── src/
│   ├── train_yolo.py
│   ├── evaluate_model.py
│   ├── detect_image.py
│   ├── detect_video.py
│   ├── analyze_formation.py
│   ├── nearest_neighbor_analysis.py
│   ├── compare_lat_std_detection_quality.py
│   └── plot_*.py
│
├── videos/
├── app.py
│
├── formation_visualizations/
├── formation_error_analysis/
│   └── ...
│
└── requirements.txt
```

## 5. Dataset Structure

Each dataset follows the standard YOLO folder structure:

```text
datasets/
└── dataset_name/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    ├── metadata/
    │
    └── data.yaml
```

Each image has a corresponding YOLO annotation file with the same filename. For example:

```text
images/train/example_001.jpg
labels/train/example_001.txt
```

The annotation files follow the standard YOLO format:

```text
class_id x_center y_center width height
```

where:

- `class_id` is the integer identifier of the object class.
- `x_center` is the normalized x-coordinate of the bounding box center.
- `y_center` is the normalized y-coordinate of the bounding box center.
- `width` is the normalized bounding box width.
- `height` is the normalized bounding box height.

All coordinate values are normalized to the range **[0, 1]** relative to the image dimensions.

## 6. Dataset Descriptions

### 6.1 Dancer Category Dataset

Dataset location:

```text
datasets/dancer_category/
```

### Classes

| Class ID | Class Name |
|:--------:|------------|
| 0 | `dance_couple` |
| 1 | `female_dancer` |
| 2 | `formation` |
| 3 | `male_dancer` |

This dataset is used to train the **Dancer Category Model**.

The model detects the different object categories that appear in DanceSport competition images, including individual dancers, dance couples, and formation scenes.

---

### 6.2 Dance Style Dataset

Dataset location:

```text
datasets/dance_style/
```

### Classes

| Class ID | Class Name |
|:--------:|------------|
| 0 | `latin` |
| 1 | `standard` |

This dataset is used to train the **Dance Style Model**.

The model classifies DanceSport images into the two competition disciplines: **Latin** and **Standard**. It serves as the basis for automatically distinguishing between the two dance styles before further analysis.

---

### 6.3 Formation Couple Dataset

Dataset location:

```text
datasets/formation_couple/
```

### Classes

| Class ID | Class Name |
|:--------:|------------|
| 0 | `couple` |

This dataset is used to train the **Formation Couple Model**.

It is the primary dataset of this project, as it enables the detection of individual couples within DanceSport formation scenes. The detected couples form the basis for the subsequent spatial analyses, including coordinate extraction, nearest-neighbor analysis, and formation quality evaluation.

---

## 7. Training the Models

Model training is performed by running:

```bash
python src/train_yolo.py
```

The dataset to be used for training is selected in `src/train_yolo.py` by setting the `SELECTED_DATASET` variable:

```python
SELECTED_DATASET = "dancer_category"
# SELECTED_DATASET = "dance_style"
# SELECTED_DATASET = "formation_couple"
```

Only **one dataset** should be selected at a time.

After training, the model weights and training results are automatically stored in:

```text
runs/detect/
```

### 7.1 Training Configuration

The training configuration is defined in the `DATASETS` dictionary inside `train_yolo.py`.

```python
DATASETS = {
    "dancer_category": {
        "data": "datasets/dancer_category/data.yaml",
        "epochs": 50,
        "imgsz": 416,
        "batch": 4,
        "name": "dancer_category_model"
    },
    "dance_style": {
        "data": "datasets/dance_style/data.yaml",
        "epochs": 50,
        "imgsz": 416,
        "batch": 4,
        "name": "dance_style_model"
    },
    "formation_couple": {
        "data": "datasets/formation_couple/data.yaml",
        "epochs": 80,
        "imgsz": 640,
        "batch": 4,
        "name": "formation_couple_model"
    }
}
```

## 7.2 Training Output

After training, the results are stored in:

```text
runs/detect/
```

### Expected output folders

```text
runs/detect/
├── dancer_category_model/
├── dance_style_model/
└── formation_couple_model/
```

### Contents of each training folder

```text
model_name/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── results.csv
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxPR_curve.png
├── BoxF1_curve.png
├── BoxP_curve.png
├── BoxR_curve.png
└── val_batch*_pred.jpg
```

The trained model weights are stored in:

```text
runs/detect/model_name/weights/best.pt
```

## 8. Evaluating the Models

Model evaluation is performed using:

```text
python src/evaluate_model.py
```

The model is selected inside `src/evaluate_model.py`:

```python
SELECTED_MODEL = "dancer_category"
# SELECTED_MODEL = "dance_style"
# SELECTED_MODEL = "formation_couple"
```

Only one model should be selected at a time.

Evaluation is performed on the test split:

```python
metrics = model.val(
    data=str(data_path),
    split="test",
    imgsz=config["imgsz"],
    project="runs/evaluation",
    name=SELECTED_MODEL,
    exist_ok=True
)
```

### 8.1 Evaluation Output

The evaluation results are stored in:

```text
runs/evaluation/
├── dancer_category/
├── dance_style/
└── formation_couple/
```

Each evaluation folder contains metrics and visualizations generated by YOLO, including:

```text
model_name/
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxPR_curve.png
├── BoxF1_curve.png
├── BoxP_curve.png
├── BoxR_curve.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
├── val_batch*_labels.jpg
├── val_batch*_pred.jpg
└── results.csv
```

The returned `metrics` object can also be used programmatically to access evaluation statistics such as mAP, precision, and recall.

### 8.2 Evaluation Metrics

The following metrics are reported:

- **Precision**
- **Recall**
- **mAP@0.5**
- **mAP@0.5:0.95**

## 9. Running Test Set Predictions

To generate prediction images on the test set, run:

```bash
python src/detect_image.py
```

The model is selected in `src/detect_image.py` by setting the `SELECTED_MODEL` variable:

```python
SELECTED_MODEL = "dancer_category"
# SELECTED_MODEL = "dance_style"
# SELECTED_MODEL = "formation_couple"
```

Only **one model** should be selected at a time.

### 9.1 Test Prediction Output

The prediction results are stored in:

```text
runs/test/
├── dancer_category/
├── dance_style/
└── formation_couple/
```

Each folder contains the test images with the predicted bounding boxes.

### Integration with the Streamlit Application

The file `src/detect_image.py` also provides the function `run_image_detection(...)`, which is imported and used by `app.py`. This allows the same image detection pipeline to be reused in the Streamlit web application.

Within the Streamlit interface, the user can select the desired model through the graphical user interface. Therefore, no code modifications are required when using the web application.

## 10. Running Video Detection

The DanceSport Detection Framework supports two different methods for video detection:

1. **Standalone video detection using the Python script**
2. **Interactive video detection through the Streamlit application**

These methods use the same detection models but store their output in different locations.

---

### 10.1 Standalone Video Detection

To perform video detection directly from the command line, run

```bash
python src/detect_video.py
```

Before executing the script, select the desired detection model inside `src/detect_video.py`:

```python
# SELECTED_MODEL = "dancer_category"
# SELECTED_MODEL = "dance_style"
SELECTED_MODEL = "formation_couple"
```

The input video is specified in the configuration dictionary:

```python
"video": "videos/example_video.mp4"
```

After processing, the generated video is automatically stored in

```text
runs/videos/
```

The output is organized according to the selected detection model:

```text
runs/videos/
├── dancer_category/
├── dance_style/
└── formation_couple/
```

This workflow is intended for standalone experiments and reproducing the detection results outside the Streamlit application.

---

### 10.2 Video Detection in the Streamlit Application

The file `src/detect_video.py` also contains the function

```python
run_video_detection(...)
```

which is imported and used by `app.py`.

Within the Streamlit application, users do **not** need to modify the source code. Instead, all settings are configured directly through the graphical user interface.

The user can

- select the desired detection model,
- upload a DanceSport video,
- adjust the confidence threshold,
- start the detection process,
- preview the processed video,
- download the generated output video.

Unlike the standalone script, the model selection and input video are chosen directly in the Streamlit interface.

After processing, the generated video is automatically stored in

```text
framework/output/videos/
```

The output is organized according to the selected detection model:

```text
framework/output/videos/
├── dancer_category/
├── dance_style/
└── formation_couple/
```

Each processed video is saved in the corresponding model directory together with a timestamp to avoid overwriting previously generated results.

This separation ensures that videos generated through the Streamlit application remain independent from the videos produced by the standalone detection script.

---

### 10.3 Summary

The output locations differ depending on how the video detection is executed.

| Detection Method | Output Directory |
|------------------|------------------|
| Standalone script (`src/detect_video.py`) | `runs/videos/` |
| Streamlit application (`app.py`) | `framework/output/videos/` |

In both cases, the output videos are automatically organized into separate folders for each detection model.

## 11. Formation Analysis Workflow

The formation analysis is based on the **Formation Couple Model** and consists of the following steps:

1. Detect all couples in a formation image.
2. Extract the bounding box of each detected couple.
3. Calculate the center point of each bounding box.
4. Store the extracted coordinates.
5. Calculate nearest-neighbor distances.
6. Evaluate the formation quality using spatial metrics.
7. Compare Latin and Standard formations.

The complete workflow is implemented in:

```text
src/analyze_formation.py
```

---

## 12. Coordinate Extraction

For each detected couple, the center point of the bounding box is calculated as:

```python
x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2
```

The extracted coordinates are stored in:

```text
formation_couple_coordinates.csv
```

These coordinates serve as the basis for all subsequent formation quality analyses, including nearest-neighbor distance calculations and statistical comparisons between Latin and Standard formations.

## 13. Nearest-Neighbor Analysis

The nearest-neighbor analysis is performed by running:

```bash
python src/nearest_neighbor_analysis.py
```

The script computes the Euclidean distance from each detected couple to its nearest neighboring couple within the same formation image.

### Generated Files

The analysis produces the following output files:

```text
nearest_neighbor_analysis_all.csv
nearest_neighbor_excluded_images.csv
nearest_neighbor_analysis_filtered.csv
nearest_neighbor_group_summary_filtered.csv
```

### 13.1 Output Files

#### `nearest_neighbor_analysis_all.csv`

Contains the nearest-neighbor distance for every detected couple in all analyzed formation images before any filtering is applied.

#### `nearest_neighbor_excluded_images.csv`

Lists the images that were excluded from the final analysis because the detected number of couples was outside the accepted range.

#### `nearest_neighbor_analysis_filtered.csv`

Contains the nearest-neighbor distances for all formation images included in the final analysis after filtering.

#### `nearest_neighbor_group_summary_filtered.csv`

Provides aggregated summary statistics for each formation group, including the number of analyzed images, mean nearest-neighbor distance, standard deviation, minimum, and maximum values.

## 14. Formation Analysis Metrics

The formation analysis calculates the following spatial metrics:

- **Average nearest-neighbor distance**
- **Standard deviation of nearest-neighbor distances**
- **Coefficient of Variation (CV)**
- **Minimum nearest-neighbor distance**
- **Maximum nearest-neighbor distance**

These metrics are used to quantify the spacing and regularity of couples within a DanceSport formation.

### 14.1 Coefficient of Variation

The **Coefficient of Variation (CV)** is used to measure the consistency of the spacing between neighboring couples.

It is calculated as:

```text
CV = standard deviation / average nearest-neighbor distance
```

### Interpretation

- **Lower CV** → More regular and consistent spacing between couples.
- **Higher CV** → Less regular spacing with greater variation between neighboring couples.

The Coefficient of Variation does not replace human judging but provides an objective quantitative measure of the spatial regularity of a formation.

## 15. Formation Detection Quality

The detection quality of the **Formation Couple Model** is evaluated by running:

```bash
python src/compare_lat_std_detection_quality.py
```

The script compares the number of annotated couples (ground truth) with the number of couples detected by the model for each formation image.

### Generated Files

```text
lat_std_detection_quality_per_image.csv
lat_std_detection_quality_summary.csv
```

- **`lat_std_detection_quality_per_image.csv`** – Contains the detection quality for each analyzed image, including the annotated and detected number of couples.
- **`lat_std_detection_quality_summary.csv`** – Provides aggregated statistics summarizing the detection quality across all analyzed formation images.

---

## 16. Streamlit Application

The graphical user interface is implemented using **Streamlit**.

Start the application with:

```bash
streamlit run app.py
```

The application consists of three main pages:

### 1. Detection

The **Detection** page allows users to:

- upload images,
- upload videos,
- browse dataset images,
- select the detection model,
- adjust the confidence threshold,
- display prediction results.

### 2. Evaluation

The **Evaluation** page provides access to the trained model performance, including:

- training and evaluation metrics,
- confusion matrices,
- Precision–Recall (PR) curves,
- F1 curves,
- test prediction images.

### 3. Formation Analysis

The **Formation Analysis** page presents the spatial analysis of DanceSport formations, including:

- nearest-neighbor statistics,
- comparison charts for Latin and Standard formations,
- formation visualizations,
- detection error analysis examples,
- demonstration videos.

## 17. Reproducing the Complete Workflow

The following steps reproduce the complete workflow, from model training to formation analysis and visualization.

### Step 1: Activate the Virtual Environment

```bash
source .venv/bin/activate
```

### Step 2: Train a Model

Open `src/train_yolo.py` and select the desired dataset:

```python
SELECTED_DATASET = "formation_couple"
```

Then start the training:

```bash
python src/train_yolo.py
```

### Step 3: Evaluate the Model

Open `src/evaluate_model.py` and select the model to evaluate:

```python
SELECTED_MODEL = "formation_couple"
```

Then run:

```bash
python src/evaluate_model.py
```

### Step 4: Perform the Formation Analysis

Execute the complete formation analysis pipeline:

```bash
python src/analyze_formation.py
python src/nearest_neighbor_analysis.py
python src/compare_lat_std_detection_quality.py
```

### Step 5: Start the Streamlit Application

Launch the graphical user interface:

```bash
streamlit run app.py
```

---

## 18. CPU Training

The project was developed and tested on **CPU-only hardware**. Depending on the dataset and model configuration, training may require several hours.

Approximate training times:

- **Dancer Category Model:** Several hours
- **Dance Style Model:** Several hours
- **Formation Couple Model:** Longer training time due to the larger input image size (640 × 640) and 80 training epochs.

### Model Weights

If the trained model weights are already available, retraining is not required. The stored `best.pt` files can be used directly for:

- model evaluation,
- image prediction,
- video detection,
- formation analysis,
- inference through the Streamlit application.

The trained weights are located in:

```text
runs/detect/model_name/weights/best.pt
```

## 19. Expected Output Structure

After running the full workflow, the output structure should look like this:

```text
runs/
├── detect/
│   ├── dancer_category_model/
│   ├── dance_style_model/
│   └── formation_couple_model/
│
├── evaluation/
│   ├── dancer_category/
│   ├── dance_style/
│   └── formation_couple/
│
├── test/
│   ├── dancer_category/
│   ├── dance_style/
│   └── formation_couple/
│
└── videos/
    ├── dancer_category/
    ├── dance_style/
    └── formation_couple/
```

## 20. Purpose of the Formation Couple Model

The **Formation Couple Model** represents the main contribution of this project.

Unlike conventional object detection models, its purpose extends beyond detecting couples in DanceSport formation images. The detected bounding boxes are used to extract the spatial coordinates of each couple, forming the basis for a quantitative analysis of formation quality.

The extracted coordinates enable the analysis of:

- distances between neighboring couples,
- spacing consistency,
- formation regularity,
- differences between Latin and Standard formations,
- detection errors caused by open Latin figures or occlusions.

These analyses provide objective spatial metrics that complement traditional human evaluation and support the overall goal of this project: applying computer vision and artificial intelligence to assist the analysis of DanceSport formations.

## 21. Reproducibility

This project was developed to ensure that all experiments can be reproduced from the provided source code, datasets, trained model weights, and analysis scripts.

## Development Environment

The framework was developed and tested using the following software environment:

| Component | Version / Specification |
|----------|--------------------------|
| Operating System | macOS |
| Python | 3.11 |
| Ultralytics | YOLO26n |
| PyTorch | Latest compatible version |
| OpenCV | Latest compatible version |
| NumPy | Latest compatible version |
| Pandas | Latest compatible version |
| Matplotlib | Latest compatible version |
| Pillow | Latest compatible version |
| Streamlit | Latest compatible version |

### Hardware

The project was developed and evaluated on the following hardware:

| Component | Specification |
|----------|---------------|
| Device | MacBook Air (2017) |
| Processor | Intel Core i5 |
| Memory | 8 GB RAM |
| Graphics | Intel HD Graphics 6000 |
| Training | CPU only |
| Inference | CPU only |

## Summary

### Reproducing the Complete Workflow

The complete workflow can be reproduced by following the steps below.

### Step 1: Create and Activate the Virtual Environment

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### Step 2: Install the Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train a Model

Select the desired dataset in:

```text
src/train_yolo.py
```

Then start the training:

```bash
python src/train_yolo.py
```

The trained model weights are stored in:

```text
runs/detect/
```

### Step 4: Evaluate the Model

Select the desired model in:

```text
src/evaluate_model.py
```

Then run:

```bash
python src/evaluate_model.py
```

The evaluation results are stored in:

```text
runs/evaluation/
```

### Step 5: Generate Test Predictions

```bash
python src/detect_image.py
```

The prediction images are stored in:

```text
runs/test/
```

### Step 6: Run Video Detection

```bash
python src/detect_video.py
```

The processed videos are stored in:

```text
runs/videos/
```

### Step 7: Perform Formation Analysis

Execute the complete formation analysis pipeline:

```bash
python src/analyze_formation.py
python src/nearest_neighbor_analysis.py
python src/compare_lat_std_detection_quality.py
```

The generated analysis results are stored in:

```text
formation_visualizations/
formation_error_analysis/
```

Additional statistical results are written to the generated CSV files.

### Step 8: Launch the Streamlit Application

Start the graphical user interface:

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## Using Pretrained Models

If pretrained model weights are already available in:

```text
runs/detect/*/weights/best.pt
```

the training step can be skipped.

In this case, the following components can be executed directly using the pretrained weights:

- Model evaluation
- Test image prediction
- Video detection
- Formation analysis
- Streamlit application

---

## Expected Results

After successfully completing all steps, the following outputs should be reproducible:

- Trained YOLO model weights
- Evaluation metrics
- Precision–Recall curves
- Confusion matrices
- Test prediction images
- Processed detection videos
- Formation coordinate data
- Nearest-neighbor analysis results
- Formation quality visualizations
- Interactive Streamlit demonstrations

This workflow enables the complete reproduction and independent verification of the experiments, analyses, and results presented in this bachelor's thesis.
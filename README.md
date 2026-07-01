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

The project was developed and tested with:

Python 3.11
Ultralytics YOLO26n
PyTorch
OpenCV
NumPy
Pandas
Matplotlib
Pillow
Streamlit

The experiments were performed on a CPU-based system:

MacBook Air 2017
Intel Core i5
8 GB RAM
Intel HD Graphics 6000
No CUDA-capable GPU

## 3. Installation

### 3.1 Create a Virtual Environment

From the project root directory:

python3 -m venv .venv

Activate the virtual environment:

source .venv/bin/activate

On Windows:

.venv\Scripts\activate

### 3.2 Install Dependencies

Install all required Python packages:

pip install -r requirements.txt

If `requirements.txt` is not available, install the most important dependencies manually:

pip install ultralytics streamlit opencv-python pandas numpy matplotlib pillow

Check whether Ultralytics YOLO is installed correctly:

python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"

## 4. Project Structure

The project uses the following structure:

bachelor_dancesport/
│
│
├── datasets/
│   ├── dancer_category/
│   ├── dance_style/
│   └── formation_couple/
|
|----formation_error_analysis
|
|----formation_visualization
|
|
|----framework/
|
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
|
── videos/
|
│──- app.py
|
│
├── formation_visualizations/
├── formation_error_analysis/
|....
|
├── requirements.txt

## 5. Dataset Structure

Each dataset follows the standard YOLO folder structure:

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
    |
    |----metadata
    │
    └── data.yaml

Each image has a corresponding YOLO label file with the same filename:

images/train/example_001.jpg
labels/train/example_001.txt

The annotation files are stored in YOLO format:

class_id x_center y_center width height

All coordinates are normalized between 0 and 1.

## 6. Dataset Descriptions

### 6.1 Dancer Category Dataset

Path:

datasets/dancer_category/

Classes:

0 dance_couple
1 female_dancer
2 formation
3 male_dancer

This dataset is used to train the **Dancer Category Model**.

### 6.2 Dance Style Dataset

Path:

datasets/dance_style/

Classes:

0 latin
1 standard

This dataset is used to train the **Dance Style Model**.

---

### 6.3 Formation Couple Dataset

Path:

datasets/formation_couple/

Classes:

0 couple

This dataset is used to train the **Formation Couple Model**.

This is the most important dataset of the project because it enables the detection of individual couples within formation scenes.

## 7. Training the Models

Training is performed using:

python src/train_yolo.py

The model to be trained is selected inside `src/train_yolo.py`:

SELECTED_DATASET = "dancer_category"
# SELECTED_DATASET = "dance_style"
# SELECTED_DATASET = "formation_couple"

Only one dataset should be selected at a time.

### 7.1 Training Configuration

The training configuration is defined in the `DATASETS` dictionary inside `train_yolo.py`.

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

### 7.2 Training Output

After training, the results are stored in:

runs/detect/

Expected output folders:

runs/detect/dancer_category_model/
runs/detect/dance_style_model/
runs/detect/formation_couple_model/

Each training folder contains:

weights/
    best.pt
    last.pt

results.png
results.csv
confusion_matrix.png
confusion_matrix_normalized.png
BoxPR_curve.png
BoxF1_curve.png
BoxP_curve.png
BoxR_curve.png
val_batch*_pred.jpg

The trained model weights are stored in:

runs/detect/model_name/weights/best.pt

## 8. Evaluating the Models

Model evaluation is performed using:

python src/evaluate_model.py

The model is selected inside `src/evaluate_model.py`:

SELECTED_MODEL = "dancer_category"
# SELECTED_MODEL = "dance_style"
# SELECTED_MODEL = "formation_couple"

Only one model should be selected at a time.

Evaluation is performed on the test split:

metrics = model.val(
    data=str(data_path),
    split="test",
    imgsz=config["imgsz"],
    project="runs/evaluation",
    name=SELECTED_MODEL,
    exist_ok=True
)

### 8.1 Evaluation Output

Evaluation results are stored in:

runs/evaluation/

Expected output folders:

runs/evaluation/dancer_category/
runs/evaluation/dance_style/
runs/evaluation/formation_couple/

Each evaluation folder contains:

evaluation_metrics.csv
confusion_matrix.png
confusion_matrix_normalized.png
BoxPR_curve.png
BoxF1_curve.png
BoxP_curve.png
BoxR_curve.png
val_batch*_pred.jpg

### 8.2 Evaluation Metrics

The following metrics are reported:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

## 9. Running Test Set Predictions

To generate prediction images on the test set, run:

python src/detect_image.py

The model is selected inside `src/detect_image.py`:

SELECTED_MODEL = "dancer_category"
# SELECTED_MODEL = "dance_style"
# SELECTED_MODEL = "formation_couple"

### 9.1 Test Prediction Output

Results are stored in:

runs/test/

Expected folders:

runs/test/dancer_category/
runs/test/dance_style/
runs/test/formation_couple/

These folders contain the test images with predicted bounding boxes.

The file also contains the function run_image_detection(...), which is imported by app.py. This allows the same image detection logic to be reused inside the Streamlit interface.

In Streamlit, the user can select the model through the graphical interface. Therefore, no code changes are required when using the web application.

## 10. Running Video Detection

Video detection is performed using:

python src/detect_video.py

The model is selected inside `src/detect_video.py`:

SELECTED_MODEL = "formation_couple"
# SELECTED_MODEL = "dancer_category"
# SELECTED_MODEL = "dance_style"

The video path is defined in the configuration dictionary:

"video": "videos/example_video.mp4"

The file also contains the function run_video_detection(...), which is imported by app.py. This function is used when a user uploads a video through the Streamlit interface.
When used through Streamlit, the processed videos are stored separately in:
framework/output/videos/

Expected Streamlit output folders:
framework/output/videos/dancer_category/
framework/output/videos/dance_style/
framework/output/videos/formation_couple/

This separation ensures that terminal-based experiments and interactive Streamlit results remain organized independently.

In the Streamlit application, the user can:

select the desired model,
upload a DanceSport video,
adjust the confidence threshold,
run video detection,
preview the processed video,
download the generated output video.
Unlike the standalone script, the Streamlit workflow does not require manual changes to SELECTED_MODEL or the video path. The model and input video are selected directly through the user interface.

### 10.1 Video Output

Generated videos are stored in:

runs/videos/

Expected folders:

runs/videos/dancer_category/
runs/videos/dance_style/
runs/videos/formation_couple/


## 11. Formation Analysis Workflow

The formation analysis is based on the **Formation Couple Model**.

The workflow is:

1. Detect all couples in a formation image.
2. Extract the bounding box of each detected couple.
3. Calculate the center point of each bounding box.
4. Store the extracted coordinates.
5. Calculate nearest-neighbor distances.
6. Evaluate formation quality using spatial metrics.
7. Compare Latin and Standard formations.

---

## 12. Coordinate Extraction

For each detected couple, the center point of the bounding box is calculated as:

x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2

The extracted coordinates are saved in:

formation_couple_coordinates.csv

These coordinates are the basis for the formation quality analysis.

## 13. Nearest-Neighbor Analysis

The nearest-neighbor analysis is performed using:

python src/nearest_neighbor_analysis.py

The script calculates the distance from each couple to its nearest neighboring couple in the same image.

Generated files:

nearest_neighbor_analysis_all.csv
nearest_neighbor_excluded_images.csv
nearest_neighbor_analysis_filtered.csv
nearest_neighbor_group_summary_filtered.csv

### 13.1 Output Files

#### `nearest_neighbor_analysis_all.csv`

Contains nearest-neighbor statistics for all analyzed formation images before filtering.

#### `nearest_neighbor_excluded_images.csv`

Contains images excluded from the final analysis because the detected number of couples was outside the accepted range.

#### `nearest_neighbor_analysis_filtered.csv`

Contains only images included in the final formation analysis.

#### `nearest_neighbor_group_summary_filtered.csv`

Contains aggregated results for each formation group.

## 14. Formation Analysis Metrics

The formation analysis calculates:

* Average nearest-neighbor distance
* Standard deviation of nearest-neighbor distances
* Coefficient of Variation (CV)
* Minimum nearest-neighbor distance
* Maximum nearest-neighbor distance

### 14.1 Coefficient of Variation

The coefficient of variation is used to describe spacing consistency:

CV = standard deviation / average nearest-neighbor distance

Interpretation:

Lower CV  = more regular spacing
Higher CV = less regular spacing

The CV does not replace human judging but provides an objective measure of spatial regularity

## 15. Formation Detection Quality

The detection quality of the Formation Couple Model is evaluated using:

python src/compare_lat_std_detection_quality.py

This script compares the number of annotated couples with the number of predicted couples.

Generated files:

lat_std_detection_quality_per_image.csv
lat_std_detection_quality_summary.csv

## 16. Streamlit Application

The graphical user interface is implemented with Streamlit.

Start the application:

streamlit run app.py

The application contains three main pages:

1. **Detection**

   * Upload images.
   * Upload Video.
   * Browse dataset images.
   * Select detection model.
   * Adjust confidence threshold.
   * Display prediction results.

2. **Evaluation**

   * Show training and evaluation metrics.
   * Display confusion matrices.
   * Display PR curves and F1 curves.
   * Show test prediction images.

3. **Formation Analysis**

   * Display nearest-neighbor statistics.
   * Show formation comparison charts.
   * Show formation visualizations.
   * Show error analysis examples.
   * Show demonstration videos.

---

## 17. Reproducing the Complete Workflow

To reproduce the complete workflow from training to analysis:

### Step 1: Activate Environment

source .venv/bin/activate

### Step 2: Train Model

Open `src/train_yolo.py` and select the dataset:

SELECTED_DATASET = "formation_couple"

Then run:

python src/train_yolo.py

### Step 3: Evaluate Model

Open `src/evaluate_model.py` and select the model:

SELECTED_MODEL = "formation_couple"

Then run:

python src/evaluate_model.py

### Step 4: Run Formation Analysis

python src/analyze_formation.py
python src/nearest_neighbor_analysis.py
python src/compare_lat_std_detection_quality.py

### Step 5: Start Streamlit

streamlit run app.py

### 18. CPU Training

The project was developed on CPU-only hardware. Training can take several hours depending on the model.

Approximate training times:

Dancer Category Model: several hours
Dance Style Model: several hours
Formation Couple Model: longer due to image size 640 and 80 epochs

### Model Weights

If the trained weights are already available, the models do not need to be retrained. Evaluation, test prediction, video detection, and Streamlit inference can be executed directly using the stored `best.pt` files.

---

## 19. Expected Output Structure

After running the full workflow, the output structure should look like this:

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

## 20. Purpose of the Formation Couple Model

The Formation Couple Model is the main contribution of this project.

The model does not only detect objects. Its detections are used to extract the spatial coordinates of couples in formation scenes.

These coordinates allow the analysis of:

* distances between couples
* spacing consistency
* formation regularity
* Latin vs. Standard differences
* detection errors caused by open Latin figures or occlusions

This directly supports the goal of using computer vision for AI-assisted ballroom dance formation analysis.

## 21. Reproducibility

This project was developed to ensure that all experiments can be reproduced from the provided source code, datasets, trained model weights, and analysis scripts.

### Development Environment

The framework was developed and tested using the following environment:

Operating System: macOS
Python Version: 3.11
Ultralytics: YOLO26
PyTorch
OpenCV
NumPy
Pandas
Matplotlib
Pillow
Streamlit

Hardware used during development:

MacBook Air (2017)
Intel Core i5
8 GB RAM
Intel HD Graphics 6000
CPU-only training and inference

## summary

### Reproducing the Complete Workflow

The complete workflow can be reproduced by executing the following steps in order.

#### Step 1: Create and Activate Virtual Environment

python3 -m venv .venv

#### Step 2: Install Dependencies

pip install -r requirements.txt

#### Step 3: Train a Model

Select the desired dataset inside:

src/train_yolo.py

and run:

python src/train_yolo.py

The trained model weights will be stored in:

runs/detect/

#### Step 4: Evaluate the Model

Select the desired model inside:

src/evaluate_model.py

and run:

python src/evaluate_model.py

Evaluation results will be stored in:

runs/evaluation/

#### Step 5: Generate Test Predictions

python src/detect_image.py

Prediction images will be stored in:

runs/test/

#### Step 6: Run Video Detection

python src/detect_video.py

Processed videos will be stored in:

runs/videos/

#### Step 7: Perform Formation Analysis

python src/analyze_formation.py
python src/nearest_neighbor_analysis.py
python src/compare_lat_std_detection_quality.py

Generated analysis results will be stored in:

formation_visualizations/
formation_error_analysis/

and in the generated CSV result files.

#### Step 8: Launch the Streamlit Application

streamlit run app.py

The application will start locally at:

http://localhost:8501

### Using Pretrained Models

If the trained model weights are already available in:

runs/detect/*/weights/best.pt

the training step can be skipped.

In this case, the evaluation scripts, test prediction scripts, video detection scripts, formation analysis tools, and the Streamlit application can be executed directly using the provided weights.

### Expected Results

When all steps are executed successfully, the following outputs should be reproducible:

- Trained YOLO model weights
- Evaluation metrics
- Precision–Recall curves
- Confusion matrices
- Test prediction images
- Processed detection videos
- Formation coordinate data
- Nearest-neighbor analysis results
- Formation quality visualizations
- Streamlit-based interactive demonstrations

This reproducibility workflow enables independent verification of all experiments and results presented in the Bachelor thesis.

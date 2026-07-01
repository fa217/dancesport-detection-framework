import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.detect_image import run_image_detection
from src.detect_video import run_video_detection


st.set_page_config(
    page_title="DanceSport Detection Framework",
    layout="wide"
)

st.title("DanceSport Detection Framework")

with st.sidebar:
    st.header("Settings")

    page = st.selectbox(
        "Page",
        ["Detection", "Evaluation", "Formation Analysis"]
    )

    model_choice = st.selectbox(
        "Choose Model",
        ["Dancer Category", "Dance Style", "Formation Couple"]
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05
    )

model_paths = {
    "Dancer Category": "runs/detect/dancer_category_model/weights/best.pt",
    "Dance Style": "runs/detect/dance_style_model/weights/best.pt",
    "Formation Couple": "runs/detect/formation_couple_model/weights/best.pt"
}

dataset_paths = {
    "Dancer Category": "datasets/dancer_category",
    "Dance Style": "datasets/dance_style",
    "Formation Couple": "datasets/formation_couple"
}

evaluation_paths = {
    "Dancer Category": "runs/detect/dancer_category_model",
    "Dance Style": "runs/detect/dance_style_model",
    "Formation Couple": "runs/detect/formation_couple_model"
}

test_evaluation_paths = {
    "Dancer Category": "runs/detect/val",
    "Dance Style": "runs/detect/val",
    "Formation Couple": "runs/detect/val2"
}


@st.cache_resource
def load_model(path):
    return YOLO(str(path))


def load_metrics_from_results_csv(run_dir):
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        return None

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    if df.empty:
        return None

    last_row = df.iloc[-1]

    metric_map = {
        "Precision": "metrics/precision(B)",
        "Recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)"
    }

    metrics = []

    for metric_name, column_name in metric_map.items():
        if column_name in df.columns:
            metrics.append({
                "Metric": metric_name,
                "Value": round(float(last_row[column_name]), 4)
            })

    if not metrics:
        return None

    return pd.DataFrame(metrics)


def load_metrics_from_json(run_dir):
    metrics_json = run_dir / "test_metrics.json"

    if not metrics_json.exists():
        return None

    with open(metrics_json, "r") as f:
        data = json.load(f)

    metrics = [
        {"Metric": "Precision", "Value": data.get("precision")},
        {"Metric": "Recall", "Value": data.get("recall")},
        {"Metric": "mAP50", "Value": data.get("mAP50")},
        {"Metric": "mAP50-95", "Value": data.get("mAP50_95")}
    ]

    return pd.DataFrame(metrics)


def predict_image(model, image, image_name, selected_class, confidence_threshold):
    image_np = np.array(image)

    results = model(
        image_np,
        imgsz=416,
        conf=confidence_threshold
    )

    result = results[0]

    detections = []
    keep_indices = []

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        if selected_class == "All" or class_name == selected_class:
            keep_indices.append(i)

            detections.append({
                "Image": image_name,
                "Class": class_name,
                "Confidence": round(conf, 2)
            })

    if selected_class != "All":
        result.boxes = result.boxes[keep_indices]

    result_image = result.plot()

    return result_image, detections


def show_image_if_exists(path, caption):
    if path.exists():
        st.image(str(path), caption=caption)
        return True

    st.warning(f"{path.name} not found.")
    return False


def show_csv_file(csv_path):
    if csv_path.exists():
        with st.expander(csv_path.name, expanded=False):
            df = pd.read_csv(csv_path)
            st.write(f"{len(df)} rows × {len(df.columns)} columns")
            st.dataframe(df, width="stretch")
    else:
        st.warning(f"{csv_path.name} not found.")


model_path = Path(model_paths[model_choice])

if not model_path.exists():
    st.error(f"Model not found: {model_path}")
    st.stop()

model = load_model(model_path)


# --------------------------------------------------
# Detection Page
# --------------------------------------------------
if page == "Detection":

    st.header("Detection")
    st.caption(f"Selected model: {model_choice}")

    selected_class = st.selectbox(
        "Filter Detection Classes",
        ["All"] + list(model.names.values()),
        key="class_filter"
    )

    input_mode = st.radio(
        "Input Source",
        ["Upload Image", "Upload Video", "Browse Dataset"],
        horizontal=True
    )

    all_detections = []

    # ==================================================
    # Upload Image
    # ==================================================
    if input_mode == "Upload Image":

        uploaded_files = st.file_uploader(
            "Upload DanceSport Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:

            for uploaded_file in uploaded_files:

                image = Image.open(uploaded_file).convert("RGB")

                st.subheader(f"Original Image: {uploaded_file.name}")
                st.image(image)

                result_image, detections, _ = run_image_detection(
                    model_path=model_path,
                    image=image,
                    image_name=uploaded_file.name,
                    selected_class=selected_class,
                    confidence_threshold=confidence_threshold,
                    imgsz=640 if model_choice == "Formation Couple" else 416
                )

                st.subheader(f"Detection Result: {uploaded_file.name}")
                st.image(result_image)

                all_detections.extend(detections)

                st.divider()

            st.subheader("Detection Summary")

            if all_detections:
                st.table(pd.DataFrame(all_detections))
            else:
                st.warning("No objects detected.")

    # ==================================================
    # Upload Video
    # ==================================================
    elif input_mode == "Upload Video":

        uploaded_video = st.file_uploader(
            "Upload DanceSport Video",
            type=["mp4", "mov", "avi"]
        )

        if uploaded_video:

            video_input_dir = Path("framework/input/videos")
            video_input_dir.mkdir(parents=True, exist_ok=True)

            input_video_path = video_input_dir / uploaded_video.name

            with open(input_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            st.subheader("Original Video")
            st.video(str(input_video_path))

            if st.button("Run Video Detection"):

                with st.spinner("Running video detection..."):

                    output_video_path = run_video_detection(
                        model_path=model_path,
                        input_video_path=input_video_path,
                        model_name=model_choice,
                        confidence_threshold=confidence_threshold,
                        imgsz=640 if model_choice == "Formation Couple" else 416
                    )

                st.success("Video detection completed.")

                st.subheader("Detection Result Video")
                st.video(str(output_video_path))

                with open(output_video_path, "rb") as video_file:
                    st.download_button(
                        label="Download Detection Video",
                        data=video_file,
                        file_name=f"detected_{output_video_path.name}",
                        mime="video/mp4"
                    )

    # ==================================================
    # Browse Dataset
    # ==================================================
    elif input_mode == "Browse Dataset":

        split = st.selectbox(
            "Choose Dataset Split",
            ["train", "val", "test"]
        )

        dataset_root = Path(dataset_paths[model_choice])
        image_dir = dataset_root / "images" / split

        if not image_dir.exists():
            st.error(f"Folder not found: {image_dir}")
            st.stop()

        image_files = sorted(
            list(image_dir.glob("*.jpg")) +
            list(image_dir.glob("*.jpeg")) +
            list(image_dir.glob("*.png"))
        )

        if not image_files:
            st.warning("No images found.")
            st.stop()

        selected_image_name = st.selectbox(
            "Choose Image",
            [img.name for img in image_files]
        )

        selected_image_path = image_dir / selected_image_name

        image = Image.open(selected_image_path).convert("RGB")

        st.subheader("Original Image")
        st.image(image)

        result_image, detections, _ = run_image_detection(
            model_path=model_path,
            image=image,
            image_name=selected_image_name,
            selected_class=selected_class,
            confidence_threshold=confidence_threshold,
            imgsz=640 if model_choice == "Formation Couple" else 416
        )

        st.subheader("Detection Result")
        st.image(result_image)

        st.subheader("Detection Summary")

        if detections:
            st.table(pd.DataFrame(detections))
        else:
            st.warning("No objects detected.")

    # ==================================================
    # Example Video
    # ==================================================
    st.markdown("---")
    st.subheader("Example Detection Video")

    video_map = {
        "Dancer Category": "runs/videos/dancer_category/STD_FORM_AV.mp4",
       "Dance Style": "runs/videos/dance_style/STD_FORM_AV_001.MP4",
       "Formation Couple": "runs/videos/formation_couple/LAT_FORM_FB_AD_AV_003.mp4"

    }

    demo_video = Path(video_map[model_choice])

    if demo_video.exists():
        st.video(str(demo_video))
    else:
        st.info("No example video available.")
# --------------------------------------------------
# Evaluation Page
# --------------------------------------------------
if page == "Evaluation":

    st.header("Model Evaluation")
    st.caption(f"Selected model: {model_choice}")

    evaluation_type = st.radio(
        "Evaluation Source",
        ["Training/Validation Results", "Test Results"],
        horizontal=True
    )

    # --------------------------------------------------
    # Select Evaluation Directory
    # --------------------------------------------------
    if evaluation_type == "Training/Validation Results":

        run_dir = Path(evaluation_paths[model_choice])

    else:

        test_evaluation_paths = {
            "Dancer Category": Path("runs/evaluation/dancer_category"),
            "Dance Style": Path("runs/evaluation/dance_style"),
            "Formation Couple": Path("runs/evaluation/formation_couple")
        }

        run_dir = test_evaluation_paths[model_choice]

    if not run_dir.exists():
        st.error(f"Evaluation folder not found: {run_dir}")
        st.stop()

    # --------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------
    st.subheader("Evaluation Metrics")

    metrics_csv = run_dir / "evaluation_metrics.csv"

    if metrics_csv.exists():

        metrics_df = pd.read_csv(metrics_csv)
        st.dataframe(metrics_df, width="stretch")

    else:

        metrics_df = load_metrics_from_results_csv(run_dir)

        if metrics_df is None:
            metrics_df = load_metrics_from_json(run_dir)

        if metrics_df is not None:
            st.dataframe(metrics_df, width="stretch")
        else:
            st.warning("No metrics file found.")
            st.info(
                "Expected evaluation_metrics.csv, results.csv, or test_metrics.json."
            )

    # --------------------------------------------------
    # Training Curves
    # --------------------------------------------------
    if evaluation_type == "Training/Validation Results":

        st.subheader("Training Results")
        show_image_if_exists(
            run_dir / "results.png",
            "Training Curves"
        )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("Confusion Matrix")

    show_image_if_exists(
        run_dir / "confusion_matrix.png",
        "Confusion Matrix"
    )

    show_image_if_exists(
        run_dir / "confusion_matrix_normalized.png",
        "Normalized Confusion Matrix"
    )

    # --------------------------------------------------
    # Precision-Recall Curve
    # --------------------------------------------------
    st.subheader("Precision-Recall Curve")

    pr_curve_paths = [
        run_dir / "BoxPR_curve.png",
        run_dir / "PR_curve.png"
    ]

    found_pr_curve = False

    for pr_curve in pr_curve_paths:
        if pr_curve.exists():
            st.image(str(pr_curve), caption=pr_curve.name)
            found_pr_curve = True
            break

    if not found_pr_curve:
        st.warning("Precision-Recall curve not found.")

    # --------------------------------------------------
    # F1-Confidence Curve
    # --------------------------------------------------
    st.subheader("F1-Confidence Curve")

    f1_curve_paths = [
        run_dir / "BoxF1_curve.png",
        run_dir / "F1_curve.png"
    ]

    found_f1_curve = False

    for f1_curve in f1_curve_paths:
        if f1_curve.exists():
            st.image(str(f1_curve), caption=f1_curve.name)
            found_f1_curve = True
            break

    if not found_f1_curve:
        st.warning("F1-confidence curve not found.")

    # --------------------------------------------------
    # Prediction Examples from Evaluation
    # --------------------------------------------------
    st.subheader("Evaluation Prediction Examples")

    pred_images = sorted(
        list(run_dir.glob("val_batch*_pred.jpg")) +
        list(run_dir.glob("test_batch*_pred.jpg"))
    )

    if pred_images:

        selected_pred = st.selectbox(
            "Choose Evaluation Prediction Example",
            [img.name for img in pred_images],
            key="evaluation_prediction_example"
        )

        selected_pred_path = run_dir / selected_pred

        st.image(
            str(selected_pred_path),
            caption=selected_pred
        )

    else:

        st.warning("No evaluation prediction example images found.")

    # --------------------------------------------------
    # Test Set Predictions
    # --------------------------------------------------
    if evaluation_type == "Test Results":

        st.subheader("Test Set Predictions")

        test_prediction_paths = {
            "Dancer Category": Path("runs/test/dancer_category"),
            "Dance Style": Path("runs/test/dance_style"),
            "Formation Couple": Path("runs/test/formation_couple")
        }

        test_dir = test_prediction_paths[model_choice]

        if test_dir.exists():

            test_images = sorted(
                list(test_dir.glob("*.jpg")) +
                list(test_dir.glob("*.jpeg")) +
                list(test_dir.glob("*.png"))
            )

            if test_images:

                selected_test_image = st.selectbox(
                    "Choose Test Prediction Image",
                    [img.name for img in test_images],
                    key="test_prediction_image"
                )

                selected_test_path = test_dir / selected_test_image

                st.image(
                    str(selected_test_path),
                    caption=selected_test_image
                )

            else:

                st.warning("No test prediction images found.")

        else:

            st.warning(f"Test prediction folder not found: {test_dir}")

# --------------------------------------------------
# Formation Analysis Page
# --------------------------------------------------
if page == "Formation Analysis":

    st.header("Formation Analysis")
    st.caption("Nearest-neighbor based formation analysis")

    # --------------------------------------------------
    # 1. Formation Analysis Summary
    # --------------------------------------------------
    st.subheader("1. Formation Analysis Summary")

    summary_path = Path("nearest_neighbor_group_summary_filtered.csv")

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        st.dataframe(summary_df, width="stretch")

        st.markdown(
            """
            This table summarizes the nearest-neighbor analysis for the relevant
            formation groups. Front-view images were excluded. Only Latin Angle View,
            Latin Top View, and Standard Angle View formations were considered.
            """
        )
    else:
        st.warning("nearest_neighbor_group_summary_filtered.csv not found.")

    # --------------------------------------------------
    # 2. Formation Comparison Charts
    # --------------------------------------------------
    st.subheader("2. Formation Comparison Charts")

    avg_nn_chart = Path("formation_comparison_avg_nn.png")
    cv_chart = Path("formation_comparison_cv.png")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.markdown("#### Average Nearest Neighbor Distance")
            st.markdown(
                """
                Average distance from each couple to its nearest neighboring couple.
                Larger values indicate more space between couples.
                """
            )

            if avg_nn_chart.exists():
                st.image(
                    str(avg_nn_chart),
                    caption="Average Nearest Neighbor Distance"
                )
            else:
                st.warning("formation_comparison_avg_nn.png not found.")

    with chart_col2:
        with st.container(border=True):
            st.markdown("#### Coefficient of Variation")
            st.markdown(
                """
                Measures how uniformly couples are distributed.
                Lower values indicate a more regular formation.
                """
            )

            if cv_chart.exists():
                st.image(
                    str(cv_chart),
                    caption="Coefficient of Variation of Nearest Neighbor Distances"
                )
            else:
                st.warning("formation_comparison_cv.png not found.")

    # --------------------------------------------------
    # 3. Detection Quality Comparison
    # --------------------------------------------------
    st.subheader("3. Detection Quality Comparison")

    quality_summary_path = Path("lat_std_detection_quality_summary.csv")
    exact_match_chart = Path("formation_exact_match_comparison.png")

    if quality_summary_path.exists():
        quality_df = pd.read_csv(quality_summary_path)

        with st.container(border=True):
            st.markdown("#### LAT AV vs STD AV Detection Quality")
            st.dataframe(quality_df, width="stretch")

            st.markdown(
                """
                This table compares the predicted number of detected couples with
                the annotated ground-truth labels. The comparison is restricted to
                Angle View images to ensure comparable viewing conditions.
                """
            )
    else:
        st.warning("lat_std_detection_quality_summary.csv not found.")

    if exact_match_chart.exists():
        with st.container(border=True):
            st.markdown("#### Exact Couple Count Match Rate")
            st.image(
                str(exact_match_chart),
                caption="Exact Couple Count Match Rate: LAT AV vs STD AV"
            )
    else:
        st.warning("formation_exact_match_comparison.png not found.")

    # --------------------------------------------------
    # 4. CSV Files
    # --------------------------------------------------
    st.subheader("4. Formation Analysis CSV Files")

    csv_files = [
        Path("formation_couple_coordinates.csv"),
        Path("nearest_neighbor_analysis_filtered.csv"),
        Path("nearest_neighbor_group_summary_filtered.csv"),
        Path("lat_std_detection_quality_per_image.csv"),
        Path("lat_std_detection_quality_summary.csv")
    ]

    for csv_path in csv_files:
        show_csv_file(csv_path)

    # --------------------------------------------------
    # 5. Image Gallery
    # --------------------------------------------------
    st.subheader("5. Formation Image Gallery")

    tab_good, tab_error = st.tabs(
        ["Final Visualizations", "Error Analysis"]
    )

    with tab_good:
        st.markdown("### Final Formation Examples")

        visualization_dir = Path("formation_visualizations")

        if visualization_dir.exists():
            visualization_images = sorted(
                list(visualization_dir.glob("*.jpg")) +
                list(visualization_dir.glob("*.png"))
            )

            if visualization_images:
                cols = st.columns(3)

                for idx, img in enumerate(visualization_images):
                    with cols[idx % 3]:
                        with st.container(border=True):
                            st.image(str(img), caption=img.name)
            else:
                st.warning("No visualization images found.")
        else:
            st.warning("formation_visualizations folder not found.")

    with tab_error:
        st.markdown("### Error Candidate Examples")

        error_dir = Path("formation_error_analysis")

        if error_dir.exists():
            error_images = sorted(
                list(error_dir.glob("*.jpg")) +
                list(error_dir.glob("*.png"))
            )

            if error_images:
                cols = st.columns(2)

                for idx, img in enumerate(error_images):
                    with cols[idx % 2]:
                        with st.container(border=True):
                            st.image(str(img), caption=img.name)
            else:
                st.warning("No error analysis images found.")
        else:
            st.warning("formation_error_analysis folder not found.")

     
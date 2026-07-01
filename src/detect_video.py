from ultralytics import YOLO
from pathlib import Path
from datetime import datetime



def run_video_detection(
    model_path,
    input_video_path,
    model_name,
    confidence_threshold=0.4,
    imgsz=416
):
    model = YOLO(str(model_path))

    safe_model_name = model_name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root = (
        Path.cwd()
        / "framework"
        / "output"
        / "videos"
        / safe_model_name
    ).resolve()

    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    model.predict(
        source=str(Path(input_video_path).resolve()),
        imgsz=imgsz,
        conf=confidence_threshold,
        save=True,
        project=str(output_root),
        name=timestamp,
        exist_ok=True
    )

    output_candidates = sorted(
        list(output_dir.glob("*.mp4")) +
        list(output_dir.glob("*.mov")) +
        list(output_dir.glob("*.avi")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not output_candidates:
        raise FileNotFoundError(
            f"Processed video was not found in: {output_dir}"
        )

    return output_candidates[0]

# --------------------------------------------------
# Standalone video detection
# --------------------------------------------------
if __name__ == "__main__":

    MODELS = {
        "dancer_category": {
            "model": "runs/detect/dancer_category_model/weights/best.pt",
            "video": "videos/example_video.mp4",
            "imgsz": 416,
            "conf": 0.4
        },

        "dance_style": {
            "model": "runs/detect/dance_style_model/weights/best.pt",
            "video": "videos/example_video.mp4",
            "imgsz": 416,
            "conf": 0.4
        },

        "formation_couple": {
            "model": "runs/detect/formation_couple_model/weights/best.pt",
            "video": "videos/STD_FORM.mov",
            "imgsz": 640,
            "conf": 0.4
        }
    }

    SELECTED_MODEL = "formation_couple"
    # SELECTED_MODEL = "dancer_category"
    # SELECTED_MODEL = "dance_style"

    config = MODELS[SELECTED_MODEL]

    model_path = Path(config["model"])
    video_path = Path(config["video"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    model = YOLO(str(model_path))

    output_root = (Path.cwd() / "runs" / "videos").resolve()

    model.predict(
    source=str(video_path.resolve()),
    imgsz=config["imgsz"],
    conf=config["conf"],
    save=True,
    project=str(output_root),
    name=SELECTED_MODEL,
    exist_ok=True
    )

    print("\nVideo detection completed.")
    print(f"Model: {SELECTED_MODEL}")
    print(f"Results saved in: runs/videos/{SELECTED_MODEL}")


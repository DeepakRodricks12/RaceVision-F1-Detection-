from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator, colors


def process_video(
    model_path: str,
    input_video_path: str,
    output_video_path: str,
    label_map: dict,
    confidence_threshold: float = 0.6,
    iou_threshold: float = 0.45,
    window_name: str = "Object Detection",
    exit_keys: list = ["q", 27],
) -> None:
    """
    Process a video file to detect and annotate objects using YOLOv8 model.

    Args:
        model_path (str): Path to the YOLOv8 model weights file.
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the annotated output video.
        label_map (dict): Dictionary to map original class names to custom names.
        confidence_threshold (float): Confidence threshold for detections (0-1).
        iou_threshold (float): IOU threshold for non-maximum suppression (0-1).
        window_name (str): Name for the display window.
        exit_keys (list): List of keys to exit the processing (can be characters or keycodes).
    """
    # Load pretrained YOLOv8 model
    model = YOLO(model_path)
    model.conf = confidence_threshold
    model.iou = iou_threshold

    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run object detection
            results = model(frame, verbose=False)
            annotator = Annotator(frame.copy())
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls)
                class_name = model.names[cls_id]
                conf = float(box.conf)

                # Apply custom label mapping
                display_name = label_map.get(class_name, class_name)
                label = f"{display_name} {conf:.2f}"
                color = colors(cls_id, bgr=True)

                # Draw bounding box and label
                annotator.box_label([x1, y1, x2, y2], label, color)

            # Write and display annotated frame
            annotated_frame = annotator.result()
            out.write(annotated_frame)

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, annotated_frame)

            # Check for exit key press
            if any(
                cv2.waitKey(1) & 0xFF == ord(key) if isinstance(key, str) else cv2.waitKey(1) & 0xFF == key
                for key in exit_keys
            ):
                break

    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configuration parameters
    CONFIG = {
        "model_path": "yolov8n.pt",
        "input_video_path": "f1_pitstop.mp4",
        "output_video_path": "output_annotated.mp4",
        "label_map": {
            "motorcycle": "F1 Car",
            "suitcase": "Tyre",
            "sports ball": "Tyre",
            "car": "F1 Car",
        },
        "confidence_threshold": 0.6,
        "iou_threshold": 0.45,
    }

    # Run the video processing
    process_video(**CONFIG)
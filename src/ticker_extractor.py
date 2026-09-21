import os
import re
import cv2

from src.video_processor import get_video_info, sample_frames
from src.text_utils import normalize_text, are_similar


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def crop_ticker(frame, bottom_percent: int = 25):
    """
    First-version ticker extraction:
    crop the bottom X percent of the video frame.

    This is deliberately simple for the POC. Later this can be replaced
    by a YOLO-based ticker detector.
    """
    height, width = frame.shape[:2]

    crop_height = int(height * bottom_percent / 100)
    y1 = max(0, height - crop_height)

    crop = frame[y1:height, 0:width]

    return crop


def preprocess_ticker(image):
    """
    Prepare ticker image for PaddleOCR.

    PaddleOCR/PaddleX expects a 3-channel image.
    We enhance the image but convert it back to BGR
    before sending it to OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale small ticker text
    gray = cv2.resize(
        gray,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC,
    )

    # Mild denoising
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Improve contrast
    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    # PaddleOCR/PaddleX expects H x W x 3
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    return processed
def process_video(
    video_path: str,
    output_dir: str,
    ocr_processor,
    interval_seconds: float = 2.0,
    bottom_percent: int = 25,
    min_confidence: float = 0.5,
    max_frames: int = 0,
    progress_callback=None,
    status_callback=None,
):
    os.makedirs(output_dir, exist_ok=True)

    info = get_video_info(video_path)
    duration = max(info["duration"], 0.001)

    results = []
    seen_texts = []

    for index, (frame_number, timestamp, frame) in enumerate(
        sample_frames(
            video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )
    ):
        if progress_callback:
            progress_callback(min(timestamp / duration, 1.0))

        if status_callback:
            status_callback(
                f"Processing frame {frame_number:,} at "
                f"{format_timestamp(timestamp)}..."
            )

        ticker_crop = crop_ticker(frame, bottom_percent=bottom_percent)

        # Save the original crop. This is the image we show the user and
        # conceptually the image being passed into OCR after preprocessing.
        raw_path = os.path.join(
            output_dir,
            f"ticker_{index:05d}_{int(timestamp):06d}.jpg",
        )
        cv2.imwrite(raw_path, ticker_crop)

        processed = preprocess_ticker(ticker_crop)

        ocr_items = ocr_processor.extract_text(
            processed,
            min_confidence=min_confidence,
        )

        if not ocr_items:
            continue

        text = " ".join(item["text"] for item in ocr_items)
        text = normalize_text(text)

        if not text:
            continue

        average_confidence = sum(
            item["confidence"] for item in ocr_items
        ) / len(ocr_items)

        # Avoid adding the same ticker repeatedly while it remains on screen.
        duplicate = any(
            are_similar(text, previous)
            for previous in seen_texts
        )

        if duplicate:
            continue

        seen_texts.append(text)

        results.append(
            {
                "timestamp": format_timestamp(timestamp),
                "timestamp_seconds": timestamp,
                "frame_number": frame_number,
                "text": text,
                "confidence": average_confidence,
                "image_path": raw_path,
            }
        )

    return results

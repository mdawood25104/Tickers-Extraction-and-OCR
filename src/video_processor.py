import cv2


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("OpenCV could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def sample_frames(video_path: str, interval_seconds: float = 2.0, max_frames: int = 0):
    """
    Yields:
        (frame_number, timestamp_seconds, frame)
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("OpenCV could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps <= 0:
        cap.release()
        raise ValueError("Could not determine the video's FPS.")

    frame_step = max(1, int(round(fps * interval_seconds)))
    frame_number = 0
    processed = 0

    while frame_number < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            break

        timestamp = frame_number / fps
        yield frame_number, timestamp, frame

        processed += 1
        if max_frames > 0 and processed >= max_frames:
            break

        frame_number += frame_step

    cap.release()

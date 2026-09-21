import os
import shutil
import streamlit as st
from src.video_processor import get_video_info
from src.ticker_extractor import process_video
from src.ocr_engine import OCRProcessor

UPLOAD_DIR = "uploads"
OUTPUT_DIR = os.path.join("output", "tickers")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(
    page_title="News Ticker OCR",
    page_icon="📰",
    layout="wide",
)

st.title("📰 News Ticker OCR")
st.write(
    "Upload a news-channel video. The app samples frames, extracts the bottom ticker "
    "region, runs OCR, removes duplicate ticker text, and shows the detected tickers."
)

with st.sidebar:
    st.header("Settings")
    interval = st.slider(
        "Sample one frame every N seconds",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )
    bottom_percent = st.slider(
        "Ticker region: bottom % of frame",
        min_value=10,
        max_value=40,
        value=25,
        step=1,
    )
    confidence = st.slider(
        "Minimum OCR confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
    )
    max_frames = st.number_input(
        "Maximum frames to process (0 = all)",
        min_value=0,
        value=0,
        step=100,
    )

uploaded = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
)

if uploaded:
    extension = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
    video_path = os.path.join(UPLOAD_DIR, f"input_video{extension}")

    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success(f"Uploaded: {uploaded.name}")

    try:
        info = get_video_info(video_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{info['duration']:.1f}s")
        c2.metric("FPS", f"{info['fps']:.2f}")
        c3.metric("Resolution", f"{info['width']} × {info['height']}")
        c4.metric("Frames", f"{info['frame_count']:,}")
    except Exception as e:
        st.error(f"Could not read video: {e}")
        st.stop()

    if st.button("🚀 Extract Tickers", type="primary", use_container_width=True):
        # Clear old output
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Loading OCR model. First run may take longer...")
            ocr = OCRProcessor()

            status.info("Processing video frames...")
            results = process_video(
                video_path=video_path,
                output_dir=OUTPUT_DIR,
                ocr_processor=ocr,
                interval_seconds=float(interval),
                bottom_percent=int(bottom_percent),
                min_confidence=float(confidence),
                max_frames=int(max_frames),
                progress_callback=lambda value: progress.progress(value),
                status_callback=lambda message: status.info(message),
            )

            progress.progress(1.0)
            status.success("Processing completed.")

            st.session_state["ticker_results"] = results

        except Exception as e:
            st.exception(e)

if "ticker_results" in st.session_state:
    results = st.session_state["ticker_results"]

    st.subheader(f"Detected Tickers ({len(results)})")

    if not results:
        st.warning(
            "No readable ticker text was detected. Try lowering the OCR confidence, "
            "increasing the bottom-region percentage, or using a clearer video."
        )
    else:
        for i, result in enumerate(results, start=1):
            with st.container(border=True):
                left, right = st.columns([1, 2])

                with left:
                    image_path = result["image_path"]
                    if os.path.exists(image_path):
                        st.image(
                            image_path,
                            caption=f"Ticker #{i} — {result['timestamp']}",
                            use_container_width=True,
                        )

                with right:
                    st.markdown(f"### Ticker #{i}")
                    st.markdown(f"**Timestamp:** `{result['timestamp']}`")
                    st.markdown(f"**Frame:** `{result['frame_number']}`")
                    st.markdown(f"**Text:** {result['text']}")

                    if result.get("confidence") is not None:
                        st.markdown(
                            f"**OCR confidence:** `{result['confidence']:.2f}`"
                        )

                    st.caption(
                        "The image above is the cropped ticker region sent to OCR."
                    )

        st.download_button(
            "⬇️ Download extracted text",
            data="\n".join(
                f"[{r['timestamp']}] {r['text']}" for r in results
            ),
            file_name="extracted_tickers.txt",
            mime="text/plain",
        )

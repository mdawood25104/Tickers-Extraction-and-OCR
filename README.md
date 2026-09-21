# News Ticker OCR 

A module for extracting news tickers from uploaded videos.

## Pipeline

```text
Video
  ↓
OpenCV
  ↓
Sample frames
  ↓
Crop bottom ticker region
  ↓
Image preprocessing
  ↓
PaddleOCR
  ↓
Text normalization
  ↓
Duplicate removal
  ↓
Streamlit dashboard
```

## Current ticker detection

The first version assumes the ticker is located in the bottom part of the video.

The default is the bottom 25%.

This is intentional: it gives you a working POC before adding a trained object detector.

## Requirements

- Windows 10/11
- Python 3.11 recommended
- Internet connection for the first package/model installation
- A news-channel video such as MP4

## Setup on Windows

Open PowerShell or Git Bash inside this project folder.

### 1. Create a virtual environment

```bash
py -3.11 -m venv venv
```

If `py -3.11` does not work:

```bash
python -m venv venv
```

### 2. Activate it

PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
venv\Scripts\activate
```

Git Bash:

```bash
source venv/Scripts/activate
```

You should see `(venv)` at the beginning of the terminal prompt.

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

PaddleOCR may download model files on its first OCR run. The first run can therefore take longer.

### 5. Run the application

```bash
streamlit run app.py
```

The browser should open the Streamlit application.

If it does not, open the local URL printed in the terminal, normally:

```text
http://localhost:8501
```

## How to use

1. Upload an MP4/MOV/AVI/MKV/WebM news video.
2. Choose how often frames should be sampled.
3. Choose what percentage of the bottom of the video should be treated as the ticker.
4. Click `Extract Tickers`.
5. Wait for processing.
6. Review the ticker image, timestamp, OCR text, and confidence.
7. Download the extracted text if required.

## Recommended first settings

For a normal 1080p news video:

```text
Sample interval: 2 seconds
Ticker region: 25%
Minimum OCR confidence: 0.50
Maximum frames: 0
```

If the ticker is very small:

```text
Sample interval: 1 second
Ticker region: 30%
```

If processing is too slow:

```text
Sample interval: 3–5 seconds
```

## Project structure

```text
news-ticker-ocr/
│
├── app.py
├── requirements.txt
├── README.md
│
├── uploads/
│
├── output/
│   └── tickers/
│
└── src/
    ├── __init__.py
    ├── video_processor.py
    ├── ticker_extractor.py
    ├── ocr_engine.py
    └── text_utils.py
```

## Important: first version vs production version

This POC uses a fixed bottom-region crop:

```text
Frame
┌───────────────────────────┐
│                           │
│       Main video          │
│                           │
│                           │
├───────────────────────────┤
│       Ticker region       │
└───────────────────────────┘
```

It does NOT yet use YOLO.

For the next version, collect screenshots from different news channels and annotate the ticker bounding box. Then train a YOLO detector with:

```text
class 0 = ticker
```

The production pipeline can then become:

```text
Video
 ↓
Frame sampling
 ↓
YOLO ticker detector
 ↓
Ticker bounding box
 ↓
Ticker crop
 ↓
Preprocessing
 ↓
PaddleOCR
 ↓
Language detection
 ↓
Text cleaning
 ↓
Duplicate/event grouping
 ↓
Database
 ↓
MAM dashboard
```

## Troubleshooting

### `python` is not recognized

Install Python 3.11 and make sure Python is added to PATH. Then restart the terminal.

### `pip install` fails for PaddlePaddle

PaddlePaddle has platform/Python-version compatibility requirements. Python 3.11 is recommended for this POC. If installation fails, check the current PaddlePaddle installation instructions for your Windows/Python combination.

### No ticker is detected

Try:

- Increase ticker region from 25% to 30–35%.
- Lower OCR confidence to 0.30–0.40.
- Use a higher-quality source video.
- Use a 1-second frame interval.
- Check that the ticker is actually near the bottom.

### OCR text is wrong

This usually means the ticker crop contains too much background or the source text is too small. Try increasing the crop percentage and using a clearer video.

### First OCR run is slow

Normal. OCR model files may need to be downloaded and initialized the first time.

### `ConvertPirAttribute2RuntimeAttribute` from PaddleOCR

This POC disables PaddleOCR oneDNN acceleration because some PaddlePaddle
Windows builds fail in the static text-detection runner when oneDNN is enabled.
If dependencies are upgraded, keep `enable_mkldnn=False` unless the installed
PaddlePaddle/PaddleOCR combination is verified with a real prediction.

## Next upgrade

The recommended next step is automatic ticker detection using YOLO.

That removes the dependency on:

```text
"ticker is always in the bottom 25%"
```

and allows the system to detect ticker boxes dynamically across different news channels.

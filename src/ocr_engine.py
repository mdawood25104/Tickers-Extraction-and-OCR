import cv2
import numpy as np


class OCRProcessor:

    def __init__(self):

        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError(
                "PaddleOCR is not installed. "
                "Run: pip install -r requirements.txt"
            ) from e

        print("Loading PaddleOCR...")

        try:
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )

        except TypeError:

            try:
                self.ocr = PaddleOCR(lang="en")

            except TypeError:

                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en"
                )

        print("PaddleOCR loaded successfully.")

    def extract_text(self, image, min_confidence=0.5):

        if image is None:
            return []

        # --------------------------------------------------
        # IMPORTANT:
        # PaddleOCR/PaddleX expects a 3-channel image.
        # --------------------------------------------------

        if len(image.shape) == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        elif len(image.shape) == 3 and image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR
            )

        elif len(image.shape) != 3:

            raise ValueError(
                f"Invalid image shape for OCR: {image.shape}"
            )

        # Make sure dtype is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # PaddleOCR
        results = self.ocr.predict(image)

        extracted = []

        for result in results:

            data = None

            # ---------------------------------------------
            # New PaddleOCR result format
            # ---------------------------------------------

            if hasattr(result, "json"):

                try:

                    data = result.json

                    if callable(data):
                        data = data()

                except Exception:
                    data = None

            # JSON string -> dictionary
            if isinstance(data, str):

                import json

                try:
                    data = json.loads(data)

                except Exception:
                    data = None

            # ---------------------------------------------
            # Parse new PaddleOCR output
            # ---------------------------------------------

            if isinstance(data, dict):

                res = data.get("res", data)

                if isinstance(res, dict):

                    texts = res.get(
                        "rec_texts",
                        []
                    )

                    scores = res.get(
                        "rec_scores",
                        []
                    )

                    for i, text in enumerate(texts):

                        try:

                            score = float(
                                scores[i]
                            ) if i < len(scores) else 1.0

                        except Exception:

                            score = 1.0

                        text = str(text).strip()

                        if (
                            text
                            and score >= min_confidence
                        ):

                            extracted.append(
                                {
                                    "text": text,
                                    "confidence": score,
                                }
                            )

                    continue

            # ---------------------------------------------
            # Older PaddleOCR output
            # ---------------------------------------------

            try:

                if isinstance(result, list):

                    for line in result:

                        if (
                            isinstance(line, (list, tuple))
                            and len(line) >= 2
                        ):

                            info = line[1]

                            if (
                                isinstance(
                                    info,
                                    (list, tuple)
                                )
                                and len(info) >= 2
                            ):

                                text = str(
                                    info[0]
                                ).strip()

                                score = float(
                                    info[1]
                                )

                                if (
                                    text
                                    and score >= min_confidence
                                ):

                                    extracted.append(
                                        {
                                            "text": text,
                                            "confidence": score,
                                        }
                                    )

            except Exception:
                pass

        return extracted
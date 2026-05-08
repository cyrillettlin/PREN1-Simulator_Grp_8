import numpy as np
import cv2


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 256
HM_SIZE  = 64


class cnnCornerDetector:
    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self._ok = True
            print(f"[CNN] Modell geladen: {model_path}")
        except Exception as e:
            print(f"[CNN] ⚠️  Konnte Modell nicht laden: {e}")
            print(f"[CNN] Fallback auf klassischen Ansatz.")
            self._ok = False

    @property
    def available(self) -> bool:
        return self._ok

    # ──────────────────────────────────────────────
    # Heatmap → Koordinaten
    # ──────────────────────────────────────────────
    def heatmap_to_coords(self, heatmaps):
        coords = []
        scale = IMG_SIZE / HM_SIZE

        for i in range(4):
            hm = heatmaps[i]

            idx = np.argmax(hm)
            y, x = np.unravel_index(idx, hm.shape)

            coords.append([x * scale, y * scale])

        return np.array(coords, dtype=np.float32)

    # ──────────────────────────────────────────────
    # Hauptfunktion
    # ──────────────────────────────────────────────
    def predict(self, image_bgr: np.ndarray,
                bounding_box: tuple) -> list:

        if not self.available:
            raise RuntimeError("CNN Modell nicht verfügbar")

        x, y, w, h = bounding_box

        # Padding
        pad = int(max(w, h) * 0.08)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image_bgr.shape[1], x + w + pad)
        y1 = min(image_bgr.shape[0], y + h + pad)

        crop = image_bgr[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape[:2]

        # ── Preprocessing (IDENTISCH ZU TRAINING!)
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        normed  = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        tensor  = np.transpose(normed, (2, 0, 1))[np.newaxis].astype(np.float32)

        try:
            # ── ONNX Inference
            outputs = self.session.run(None, {self.input_name: tensor})[0]

            # Erwartet: (1,4,64,64)
            if outputs.ndim != 4:
                raise ValueError(f"Unerwartete Output-Shape: {outputs.shape}")

            heatmaps = outputs[0]  # (4,64,64)

            # 🔥 WICHTIG: Sigmoid (wegen BCEWithLogitsLoss)
            heatmaps = 1.0 / (1.0 + np.exp(-heatmaps))

            # Optional Debug:
            # print("Heatmap max:", heatmaps.max())

            # ── Koordinaten extrahieren
            coords = self.heatmap_to_coords(heatmaps)

            # ── Zurückskalieren auf Crop + Vollbild
            corners = []
            for (cx, cy) in coords:
                cx = cx * (crop_w / IMG_SIZE) + x0
                cy = cy * (crop_h / IMG_SIZE) + y0
                corners.append((int(cx), int(cy)))

            return corners

        except Exception as e:
            print(f"[CNN] Inference Fehler: {e}")
            raise

    # ──────────────────────────────────────────────
    # Kontur-Snapping
    # ──────────────────────────────────────────────
    def predict_on_contour(self, image_bgr: np.ndarray,
                           bounding_box: tuple,
                           contour: np.ndarray) -> list:

        raw_corners = self.predict(image_bgr, bounding_box)

        pts = contour.reshape(-1, 2).astype(np.float32)

        snapped = []
        for (cx, cy) in raw_corners:
            dists   = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
            nearest = pts[np.argmin(dists)].astype(int)
            snapped.append(tuple(nearest))

        return snapped
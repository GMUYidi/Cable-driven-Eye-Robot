from __future__ import annotations

from pathlib import Path


class StillFrameCapture:
    def __init__(self, name: str, url: str, warmup_frames: int = 5):
        self.name = name
        self.url = url
        self.warmup_frames = warmup_frames
        self.cap = None

    def open(self) -> None:
        import cv2

        self.cap = cv2.VideoCapture(self.url)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {self.name} camera stream: {self.url}")

    def capture(self, output_path: Path) -> None:
        import cv2

        if self.cap is None:
            self.open()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame = None
        for _ in range(max(1, self.warmup_frames)):
            ok, candidate = self.cap.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            raise RuntimeError(f"{self.name} camera did not return a valid frame")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write image: {output_path}")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

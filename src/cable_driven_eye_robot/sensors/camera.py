from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CameraFrame:
    frame: object
    fps: float
    timestamp_s: float


class CameraStream(threading.Thread):
    def __init__(
        self,
        name: str,
        url: str,
        callback: Callable[[str, CameraFrame], None] | None = None,
        reconnect_delay_s: float = 2.0,
    ):
        super().__init__(daemon=True)
        self.name = name
        self.url = url
        self.callback = callback
        self.reconnect_delay_s = reconnect_delay_s
        self.stop_event = threading.Event()
        self.last_error = ""

    def close(self) -> None:
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=1.0)

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.last_error = "opencv-python is not installed"
            return

        while not self.stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.url)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if not cap.isOpened():
                    self.last_error = f"Could not open camera stream: {self.url}"
                    time.sleep(self.reconnect_delay_s)
                    continue

                frame_count = 0
                fps = 0.0
                last_fps_time = time.monotonic()
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self.last_error = "camera frame read failed"
                        break

                    frame_count += 1
                    now = time.monotonic()
                    if now - last_fps_time >= 1.0:
                        fps = frame_count / max(now - last_fps_time, 1e-6)
                        frame_count = 0
                        last_fps_time = now

                    if self.callback is not None:
                        self.callback(self.name, CameraFrame(frame, fps, now))
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(self.reconnect_delay_s)
            finally:
                if cap is not None:
                    cap.release()

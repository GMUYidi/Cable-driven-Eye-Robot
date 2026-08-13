from __future__ import annotations

import argparse
import ctypes
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from replay_healthbaseline_twoeyes import (
    DEFAULT_CAMERA_BAUD,
    DEFAULT_FRAME_DIR,
    DEFAULT_LEFT_CAMERA_PORT,
    DEFAULT_LEFT_SAVE_DIR,
    DEFAULT_RIGHT_CAMERA_PORT,
    DEFAULT_RIGHT_SAVE_DIR,
    SerialCamera,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def force_window_to_front(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    SW_SHOW = 5
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040

    user32.ShowWindow(hwnd, SW_SHOW)
    user32.SetWindowPos(
        hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
    )
    user32.SetForegroundWindow(hwnd)
    user32.SetActiveWindow(hwnd)
    user32.SetFocus(hwnd)
    user32.SetWindowPos(
        hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
    )


class StaticFullscreenFrame:
    def __init__(self, image_path: Path) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("First frame capture")
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        self.size = self.screen.get_size()
        self.hwnd = pygame.display.get_wm_info().get("window")

        image = pygame.image.load(str(image_path)).convert()
        self.image = pygame.transform.smoothscale(image, self.size)
        self.closed = False
        self.draw()
        if self.hwnd:
            force_window_to_front(int(self.hwnd))

    def draw(self) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.closed = True
            elif event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                self.closed = True

        if self.closed:
            return False

        self.screen.blit(self.image, (0, 0))
        self.pygame.display.flip()
        return True

    def pump_for(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not self.draw():
                return False
            time.sleep(0.03)
        return True

    def close(self) -> None:
        self.pygame.quit()


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    key.append(path.suffix.lower())
    return key


def first_frame_path(frame_dir: Path) -> Path:
    preferred = frame_dir / "0.jpg"
    if preferred.exists():
        return preferred

    images = [
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No image files found in {frame_dir}")
    return sorted(images, key=natural_key)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the first frame fullscreen and capture both eye cameras."
    )
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--left-save-dir", type=Path, default=DEFAULT_LEFT_SAVE_DIR)
    parser.add_argument("--right-save-dir", type=Path, default=DEFAULT_RIGHT_SAVE_DIR)
    parser.add_argument("--left-camera-port", default=DEFAULT_LEFT_CAMERA_PORT)
    parser.add_argument("--right-camera-port", default=DEFAULT_RIGHT_CAMERA_PORT)
    parser.add_argument("--camera-baud", type=int, default=DEFAULT_CAMERA_BAUD)
    parser.add_argument("--display-settle-s", type=float, default=0.5)
    parser.add_argument("--hold-after-capture-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    frame_path = first_frame_path(args.frame_dir)
    args.left_save_dir.mkdir(parents=True, exist_ok=True)
    args.right_save_dir.mkdir(parents=True, exist_ok=True)

    left_output = args.left_save_dir / "first_frame_left.jpg"
    right_output = args.right_save_dir / "first_frame_right.jpg"

    presenter: StaticFullscreenFrame | None = None
    left_camera: SerialCamera | None = None
    right_camera: SerialCamera | None = None

    try:
        left_camera = SerialCamera(args.left_camera_port, args.camera_baud)
        right_camera = SerialCamera(args.right_camera_port, args.camera_baud)
        left_camera.open()
        right_camera.open()

        presenter = StaticFullscreenFrame(frame_path)
        if not presenter.pump_for(args.display_settle_s):
            print("Fullscreen display was closed before capture.")
            return 1

        with ThreadPoolExecutor(max_workers=2) as pool:
            left_future = pool.submit(left_camera.capture_to, left_output)
            right_future = pool.submit(right_camera.capture_to, right_output)

            while not (left_future.done() and right_future.done()):
                if not presenter.draw():
                    print("Fullscreen display was closed during capture.")
                    return 1
                time.sleep(0.03)

            left_future.result()
            right_future.result()

        print(f"Displayed frame: {frame_path}")
        print(f"Saved left image: {left_output}")
        print(f"Saved right image: {right_output}")

        presenter.pump_for(args.hold_after_capture_s)
        return 0

    finally:
        if left_camera is not None:
            left_camera.close()
        if right_camera is not None:
            right_camera.close()
        if presenter is not None:
            presenter.close()


if __name__ == "__main__":
    raise SystemExit(main())

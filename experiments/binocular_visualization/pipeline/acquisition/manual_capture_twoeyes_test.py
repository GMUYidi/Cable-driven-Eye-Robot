from __future__ import annotations

from pathlib import Path

from replay_healthbaseline_twoeyes import (
    DEFAULT_CAMERA_BAUD,
    DEFAULT_LEFT_CAMERA_PORT,
    DEFAULT_RIGHT_CAMERA_PORT,
    SerialCamera,
    capture_both_cameras,
)


SAVE_DIR = Path("D:/visualization2026/test")
LEFT_IMAGE_PATH = SAVE_DIR / "left.jpg"
RIGHT_IMAGE_PATH = SAVE_DIR / "right.jpg"


def main() -> int:
    left_camera = SerialCamera(DEFAULT_LEFT_CAMERA_PORT, DEFAULT_CAMERA_BAUD)
    right_camera = SerialCamera(DEFAULT_RIGHT_CAMERA_PORT, DEFAULT_CAMERA_BAUD)

    print(f"Opening left camera on {DEFAULT_LEFT_CAMERA_PORT}...")
    left_camera.open()
    print(f"Opening right camera on {DEFAULT_RIGHT_CAMERA_PORT}...")
    right_camera.open()
    print("Ready. Type 'capture' to save left/right images, or 'quit' to exit.")

    try:
        while True:
            command = input("> ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break
            if command != "capture":
                print("Type 'capture' to take photos, or 'quit' to exit.")
                continue

            left_path, right_path = capture_both_cameras(
                left_camera,
                right_camera,
                LEFT_IMAGE_PATH,
                RIGHT_IMAGE_PATH,
            )
            print(f"Saved left:  {left_path}")
            print(f"Saved right: {right_path}")
    finally:
        left_camera.close()
        right_camera.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

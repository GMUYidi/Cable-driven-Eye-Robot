from __future__ import annotations

import math
import time
import tkinter as tk

from ..math_utils import clamp
from .state import CameraDisplayState, DashboardSnapshot, EyeDisplayState, SharedRobotState


class EyeCameraDashboard:
    def __init__(
        self,
        root: tk.Tk,
        state: SharedRobotState,
        stop_callback,
        yaw_gain: float = 1.65,
        pitch_gain: float = 1.65,
        roll_gain: float = 1.65,
    ):
        self.root = root
        self.state = state
        self.stop_callback = stop_callback
        self.yaw_gain = yaw_gain
        self.pitch_gain = pitch_gain
        self.roll_gain = roll_gain
        self.photo_refs = {}
        self.last_frame_time = time.time()

        self.root.title("Cable-Driven Eye Robot Dashboard")
        self.root.configure(bg="#eef3f8")
        self.canvas = tk.Canvas(root, width=1280, height=900, bg="#eef3f8", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.bind("<q>", lambda _event: self.request_close())
        self.root.bind("<Escape>", lambda _event: self.request_close())
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        self.draw()

    def request_close(self) -> None:
        self.stop_callback()
        self.root.destroy()

    def frame_alpha(self) -> float:
        now = time.time()
        dt = clamp(now - self.last_frame_time, 0.001, 0.1)
        self.last_frame_time = now
        return 1.0 - math.exp(-dt * 12.0)

    def draw(self) -> None:
        snapshot = self.state.snapshot(alpha=self.frame_alpha())
        self.canvas.delete("all")

        width = max(self.canvas.winfo_width(), 900)
        height = max(self.canvas.winfo_height(), 700)
        self.draw_background(width, height)

        top_area_h = height * 0.54
        eye_radius = min(width * 0.12, top_area_h * 0.26, 125)
        eye_y = top_area_h * 0.45
        spacing = eye_radius * 1.55
        self.draw_eye(width / 2 - spacing, eye_y, eye_radius, snapshot.left_eye, "LEFT IMU")
        self.draw_eye(width / 2 + spacing, eye_y, eye_radius, snapshot.right_eye, "RIGHT IMU")
        self.draw_control_hud(width, top_area_h, snapshot)

        cam_top = int(top_area_h)
        footer_h = 92
        gap = 16
        margin = 18
        panel_w = (width - margin * 2 - gap) / 2
        panel_h = min(height - cam_top - footer_h - margin, panel_w * 0.76)
        panel_h = max(panel_h, 170)
        self.draw_camera_panel(margin, cam_top + 8, panel_w, panel_h, snapshot.left_camera, "LEFT_CAMERA")
        self.draw_camera_panel(
            margin + panel_w + gap,
            cam_top + 8,
            panel_w,
            panel_h,
            snapshot.right_camera,
            "RIGHT_CAMERA",
        )
        self.draw_logs(width, height, footer_h, snapshot.logs)
        self.root.after(16, self.draw)

    def draw_background(self, width: int, height: int) -> None:
        self.canvas.create_rectangle(0, 0, width, height, fill="#eef3f8", outline="")
        self.canvas.create_rectangle(0, 0, width, height * 0.54, fill="#e7f0f7", outline="")
        self.canvas.create_text(
            width / 2,
            28,
            text="Cable-Driven Eye Robot",
            fill="#1f2937",
            font=("Segoe UI", 22, "bold"),
        )
        self.canvas.create_text(
            width / 2,
            58,
            text="Joystick open-loop control with dual IMU and dual ESP32 camera streams",
            fill="#52606d",
            font=("Segoe UI", 11),
        )

    def draw_eye(self, cx: float, cy: float, radius: float, state: EyeDisplayState, label: str) -> None:
        yaw = clamp(state.yaw_deg * self.yaw_gain, -60.0, 60.0)
        pitch = clamp(state.pitch_deg * self.pitch_gain, -45.0, 45.0)
        roll = state.roll_deg * self.roll_gain

        gaze_x = -(yaw / 60.0) * radius * 0.43
        gaze_y = (pitch / 45.0) * radius * 0.36
        iris_r = radius * 0.30
        pupil_r = radius * 0.13

        stale = state.has_data and (time.time() - state.last_update_s > 1.5)
        outline = "#27ae60" if state.connected and not stale else "#d35400"
        if not state.has_data:
            outline = "#8b98a5"

        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill="#ffffff", outline="#c8d1dc", width=4)
        iris_x = cx + gaze_x
        iris_y = cy + gaze_y
        self.canvas.create_oval(iris_x - iris_r, iris_y - iris_r, iris_x + iris_r, iris_y + iris_r, fill="#4aa3df", outline="#1f6f9f", width=3)
        self.canvas.create_oval(iris_x - pupil_r, iris_y - pupil_r, iris_x + pupil_r, iris_y + pupil_r, fill="#111827", outline="")

        roll_rad = math.radians(roll)
        for angle_deg in (0, 90, 180, 270):
            angle = roll_rad + math.radians(angle_deg)
            x1 = iris_x + math.cos(angle) * iris_r * 0.20
            y1 = iris_y + math.sin(angle) * iris_r * 0.20
            x2 = iris_x + math.cos(angle) * iris_r * 0.85
            y2 = iris_y + math.sin(angle) * iris_r * 0.85
            self.canvas.create_line(x1, y1, x2, y2, fill="#8ecbf1", width=2)

        self.canvas.create_oval(cx - radius * 1.04, cy - radius * 1.04, cx + radius * 1.04, cy + radius * 1.04, outline=outline, width=3)
        self.canvas.create_text(cx, cy + radius + 28, text=label, fill="#1f2937", font=("Segoe UI", 14, "bold"))
        if state.has_data:
            text = f"yaw {state.target_yaw_deg:6.1f}   pitch {state.target_pitch_deg:6.1f}   roll {state.target_roll_deg:6.1f}"
        else:
            text = "waiting for IMU serial data..."
        self.canvas.create_text(cx, cy + radius + 52, text=text, fill="#374151", font=("Consolas", 10))

    def draw_control_hud(self, width: int, top_area_h: float, snapshot: DashboardSnapshot) -> None:
        control = snapshot.control
        x0 = 22
        y0 = top_area_h - 96
        x1 = width - 22
        y1 = top_area_h - 14
        self.canvas.create_rectangle(x0, y0, x1, y1, fill="#1f2937", outline="")
        text = (
            f"Joystick: {control.joystick_name or 'not detected'}    "
            f"DXL: {control.message}    "
            f"yaw {control.yaw_cmd_deg:+5.2f} deg    "
            f"pitch {control.pitch_cmd_deg:+5.2f} deg    "
            f"roll {control.roll_cmd_deg:+5.2f} deg"
        )
        self.canvas.create_text(x0 + 14, y0 + 24, anchor="w", text=text, fill="#e5edf5", font=("Segoe UI", 11, "bold"))

    def draw_camera_panel(self, x: float, y: float, w: float, h: float, state: CameraDisplayState, label: str) -> None:
        self.canvas.create_rectangle(x, y, x + w, y + h, fill="#111820", outline="#344554", width=2)
        title_h = 30
        self.canvas.create_rectangle(x, y, x + w, y + title_h, fill="#1f2937", outline="")
        status = "connected" if state.connected else "waiting"
        self.canvas.create_text(x + 12, y + title_h / 2, anchor="w", text=f"{label.replace('_', ' ')}  {status}  {state.fps:4.1f} fps", fill="#e5edf5", font=("Segoe UI", 11, "bold"))

        content_x = x + 8
        content_y = y + title_h + 8
        content_w = w - 16
        content_h = h - title_h - 16
        self.canvas.create_rectangle(content_x, content_y, content_x + content_w, content_y + content_h, fill="#05070a", outline="")

        try:
            import cv2
            from PIL import Image, ImageTk
        except ImportError:
            self.canvas.create_text(x + w / 2, content_y + content_h / 2, text="install opencv-python and pillow for camera display", fill="#9fb0bf", font=("Segoe UI", 10))
            return

        if state.frame is None:
            self.canvas.create_text(x + w / 2, content_y + content_h / 2, text=state.last_error or "waiting for stream", fill="#9fb0bf", font=("Segoe UI", 11))
            return

        try:
            frame = cv2.rotate(state.frame, cv2.ROTATE_180)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((int(content_w), int(content_h)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image, master=self.root)
            self.photo_refs[label] = photo
            self.canvas.create_image(x + w / 2, content_y + content_h / 2, image=self.photo_refs[label], anchor="center")
        except Exception as exc:
            self.canvas.create_text(x + w / 2, content_y + content_h / 2, text=f"frame render error: {exc}", fill="#fca5a5", font=("Segoe UI", 10))

    def draw_logs(self, width: int, height: int, footer_h: int, logs: list[str]) -> None:
        self.canvas.create_rectangle(0, height - footer_h, width, height, fill="#101820", outline="")
        y = height - footer_h + 16
        for log in logs[-4:]:
            self.canvas.create_text(18, y, anchor="w", text=log, fill="#b9c6d3", font=("Consolas", 9))
            y += 18

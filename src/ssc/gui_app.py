from __future__ import annotations

import os
import threading
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ssc.config import MotionConfig, RecorderConfig, HOG_TUNING_DEFAULTS
from ssc.features.dnn_person_detector import find_default_dnn_model_path
from ssc.pipeline import process_video


def _default_dnn_model_path() -> str:
    project_root = Path(__file__).resolve().parents[2]
    default_path = find_default_dnn_model_path(project_root)
    return str(default_path) if default_path is not None else ""


class App:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("CSY3058 Smart Security Camera")

        # keeping the GUI state in Tk variables so widgets update nicely

        self.input_path = StringVar()
        self.output_dir = StringVar(value=str(Path("outputs").resolve()))
        self.min_area = IntVar(value=HOG_TUNING_DEFAULTS.min_contour_area)
        self.post_frames = IntVar(value=30)
        self.close_iters = IntVar(value=HOG_TUNING_DEFAULTS.morph_close_iterations)
        self.dilate_iters = IntVar(value=HOG_TUNING_DEFAULTS.dilate_iterations)
        self.merge_pad = IntVar(value=HOG_TUNING_DEFAULTS.merge_bbox_padding)
        self.learning_rate = DoubleVar(value=HOG_TUNING_DEFAULTS.mog2_learning_rate)
        self.on_frames = IntVar(value=HOG_TUNING_DEFAULTS.motion_on_frames)
        self.off_frames = IntVar(value=HOG_TUNING_DEFAULTS.motion_off_frames)
        self.roi_rects = StringVar(value="")
        self.prob_draw = BooleanVar(value=True)
        self.single_box = BooleanVar(value=True)
        self.use_cc = BooleanVar(value=True)
        self.display = BooleanVar(value=False)
        self.show_mask = BooleanVar(value=False)
        self.person_detector = StringVar(value="motion")

        self.status = StringVar(value="idle")

        self._widgets_to_disable: list[ttk.Widget] = []
        self._progress: ttk.Progressbar | None = None
        self._stop_event = threading.Event()
        self.stop_btn: ttk.Button | None = None

        self._build()

    def _build(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        input_box = ttk.LabelFrame(container, text="Input", padding=10)
        input_box.grid(row=0, column=0, sticky="ew")
        input_box.columnconfigure(1, weight=1)

        ttk.Label(input_box, text="Video file").grid(row=0, column=0, sticky="w")
        input_entry = ttk.Entry(input_box, textvariable=self.input_path)
        input_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        input_browse = ttk.Button(input_box, text="Browse", command=self._browse_input)
        input_browse.grid(row=0, column=2)

        ttk.Label(input_box, text="Output folder").grid(row=1, column=0, sticky="w", pady=(8, 0))
        out_entry = ttk.Entry(input_box, textvariable=self.output_dir)
        out_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        out_browse = ttk.Button(input_box, text="Browse", command=self._browse_output)
        out_browse.grid(row=1, column=2, pady=(8, 0))

        settings_box = ttk.LabelFrame(container, text="Settings", padding=10)
        settings_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        settings_box.columnconfigure(1, weight=1)

        ttk.Label(settings_box, text="min-area").grid(row=0, column=0, sticky="w")
        min_area_spin = ttk.Spinbox(settings_box, from_=50, to=50000, textvariable=self.min_area, width=12)
        min_area_spin.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(settings_box, text="post-frames").grid(row=1, column=0, sticky="w", pady=(8, 0))
        post_spin = ttk.Spinbox(settings_box, from_=0, to=500, textvariable=self.post_frames, width=12)
        post_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="close-iters").grid(row=2, column=0, sticky="w", pady=(8, 0))
        close_spin = ttk.Spinbox(settings_box, from_=0, to=10, textvariable=self.close_iters, width=12)
        close_spin.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="dilate-iters").grid(row=3, column=0, sticky="w", pady=(8, 0))
        dilate_spin = ttk.Spinbox(settings_box, from_=0, to=10, textvariable=self.dilate_iters, width=12)
        dilate_spin.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="merge-pad").grid(row=4, column=0, sticky="w", pady=(8, 0))
        merge_spin = ttk.Spinbox(settings_box, from_=0, to=100, textvariable=self.merge_pad, width=12)
        merge_spin.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="learning-rate").grid(row=5, column=0, sticky="w", pady=(8, 0))
        lr_spin = ttk.Spinbox(settings_box, from_=-1.0, to=1.0, increment=0.01, textvariable=self.learning_rate, width=12)
        lr_spin.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="on-frames").grid(row=6, column=0, sticky="w", pady=(8, 0))
        on_spin = ttk.Spinbox(settings_box, from_=1, to=30, textvariable=self.on_frames, width=12)
        on_spin.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="off-frames").grid(row=7, column=0, sticky="w", pady=(8, 0))
        off_spin = ttk.Spinbox(settings_box, from_=1, to=60, textvariable=self.off_frames, width=12)
        off_spin.grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="roi (x1,y1,x2,y2;...)").grid(row=8, column=0, sticky="w", pady=(8, 0))
        roi_entry = ttk.Entry(settings_box, textvariable=self.roi_rects)
        roi_entry.grid(row=8, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        roi_btn = ttk.Button(settings_box, text="Select ROI...", command=self._pick_roi)
        roi_btn.grid(row=8, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="person-detector").grid(row=9, column=0, sticky="w", pady=(8, 0))
        detector_combo = ttk.Combobox(
            settings_box,
            textvariable=self.person_detector,
            state="readonly",
            values=["motion", "hog", "dnn"],
            width=12,
        )
        detector_combo.grid(row=9, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        single_cb = ttk.Checkbutton(settings_box, text="Single box mode", variable=self.single_box)
        single_cb.grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 0))

        cc_cb = ttk.Checkbutton(settings_box, text="Use connected components", variable=self.use_cc)
        cc_cb.grid(row=11, column=0, columnspan=2, sticky="w")

        prob_cb = ttk.Checkbutton(settings_box, text="Probabilistic smoothing (draw only)", variable=self.prob_draw)
        prob_cb.grid(row=12, column=0, columnspan=2, sticky="w")

        display_cb = ttk.Checkbutton(settings_box, text="Display preview (OpenCV)", variable=self.display)
        display_cb.grid(row=13, column=0, columnspan=2, sticky="w")
        mask_cb = ttk.Checkbutton(settings_box, text="Show motion mask", variable=self.show_mask)
        mask_cb.grid(row=14, column=0, columnspan=2, sticky="w")

        actions_box = ttk.Frame(container)
        actions_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        actions_box.columnconfigure(0, weight=1)

        self.run_btn = ttk.Button(actions_box, text="Run detection", command=self._run)
        self.run_btn.grid(row=0, column=0, sticky="ew")

        self.stop_btn = ttk.Button(actions_box, text="Stop / Cancel", command=self._cancel, state="disabled")
        self.stop_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        open_btn = ttk.Button(actions_box, text="Open outputs folder", command=self._open_output_folder)
        open_btn.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        status_box = ttk.LabelFrame(container, text="Status", padding=10)
        status_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        status_box.columnconfigure(0, weight=1)

        self._progress = ttk.Progressbar(status_box, mode="indeterminate")
        self._progress.grid(row=0, column=0, sticky="ew")

        ttk.Label(status_box, textvariable=self.status).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self._widgets_to_disable = [
            input_entry,
            input_browse,
            out_entry,
            out_browse,
            min_area_spin,
            post_spin,
            close_spin,
            dilate_spin,
            merge_spin,
            lr_spin,
            on_spin,
            off_spin,
            roi_entry,
            roi_btn,
            detector_combo,
            single_cb,
            cc_cb,
            prob_cb,
            display_cb,
            mask_cb,
            open_btn,
        ]

    def _pick_roi(self) -> None:
        inp = Path(self.input_path.get()).expanduser()
        if not inp.exists():
            messagebox.showerror("Missing input", "Please select a valid video file first.")
            return

        cap = cv2.VideoCapture(str(inp))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            messagebox.showerror("ROI", "Could not read first frame from the selected video.")
            return

        rois: list[tuple[int, int, int, int]] = []
        while True:
            x, y, w, h = cv2.selectROI("Select ROI (ENTER=accept, ESC=cancel)", frame, showCrosshair=True)
            if w == 0 or h == 0:
                break
            rois.append((int(x), int(y), int(x + w), int(y + h)))

            keep = messagebox.askyesno("ROI", "Add another ROI?")
            if not keep:
                break

        cv2.destroyWindow("Select ROI (ENTER=accept, ESC=cancel)")

        if rois:
            self.roi_rects.set("; ".join(f"{x1},{y1},{x2},{y2}" for x1, y1, x2, y2 in rois))

    def _browse_input(self) -> None:
        initial = None
        try:
            default = Path(__file__).resolve().parents[3] / "data"
            if default.exists():
                initial = str(default)
        except Exception:
            initial = None

        path = filedialog.askopenfilename(
            title="Select a video file",
            initialdir=initial,
            filetypes=[("Video files", "*.avi *.mp4"), ("All files", "*.*")],
        )
        if path:
            self.input_path.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)


    def _open_output_folder(self) -> None:
        try:
            out = Path(self.output_dir.get()).expanduser().resolve()
            out.mkdir(parents=True, exist_ok=True)

            if os.name == "nt":
                os.startfile(str(out))
            else:
                messagebox.showinfo("Output folder", str(out))
        except Exception as e:
            messagebox.showerror("Could not open folder", str(e))

    def _set_running(self, running: bool) -> None:
        self.run_btn.configure(state=("disabled" if running else "normal"))
        if self.stop_btn is not None:
            self.stop_btn.configure(state=("normal" if running else "disabled"))
        for w in self._widgets_to_disable:
            w.configure(state=("disabled" if running else "normal"))

        if self._progress is not None:
            if running:
                self._progress.start(10)
            else:
                self._progress.stop()

    def _run(self) -> None:
        inp = Path(self.input_path.get()).expanduser()
        out = Path(self.output_dir.get()).expanduser()

        if not inp.exists():
            messagebox.showerror("Missing input", "Please select a valid video file.")
            return

        out.mkdir(parents=True, exist_ok=True)

        roi_rects: list[tuple[int, int, int, int]] = []
        raw = self.roi_rects.get().strip()
        if raw:
            for item in raw.split(";"):
                s = item.strip()
                if not s:
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) != 4:
                    messagebox.showerror("Invalid ROI", "ROI must be x1,y1,x2,y2;... (integers)")
                    return
                try:
                    x1, y1, x2, y2 = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
                except ValueError:
                    messagebox.showerror("Invalid ROI", "ROI must be x1,y1,x2,y2;... (integers)")
                    return
                roi_rects.append((x1, y1, x2, y2))

        motion_cfg = MotionConfig(
            min_contour_area=int(self.min_area.get()),
            morph_close_iterations=int(self.close_iters.get()),
            dilate_iterations=int(self.dilate_iters.get()),
            merge_bbox_padding=int(self.merge_pad.get()),
            mog2_learning_rate=float(self.learning_rate.get()),
            use_connected_components=bool(self.use_cc.get()),
            single_box=bool(self.single_box.get()),
            motion_on_frames=int(self.on_frames.get()),
            motion_off_frames=int(self.off_frames.get()),
            prob_presence_draw=bool(self.prob_draw.get()),
            roi_rects=tuple(roi_rects),
        )
        recorder_cfg = RecorderConfig(post_event_frames=int(self.post_frames.get()))

        detector_mode = (self.person_detector.get() or "motion").strip().lower()
        if detector_mode not in ("motion", "hog", "dnn"):
            detector_mode = "motion"

        dnn_path_text = _default_dnn_model_path()
        dnn_model_path = Path(dnn_path_text).expanduser() if dnn_path_text else None
        if detector_mode == "dnn":
            if (
                dnn_model_path is None
                or not dnn_model_path.is_file()
                or not dnn_model_path.with_suffix(".prototxt").is_file()
                or not dnn_model_path.with_suffix(".caffemodel").is_file()
            ):
                messagebox.showerror(
                    "Missing DNN Model",
                    "Default MobileNet-SSD model not found in models/.",
                )
                return

        self.status.set("running...")
        self._stop_event.clear()
        self._set_running(True)

        def worker() -> None:
            # run in a thread so the UI doesn't freeze while processing frames
            try:
                process_video(
                    input_path=inp,
                    output_dir=out,
                    motion_cfg=motion_cfg,
                    recorder_cfg=recorder_cfg,
                    display=bool(self.display.get()),
                    show_mask=bool(self.show_mask.get()),
                    stop_event=self._stop_event,
                    person_detector=detector_mode,
                    dnn_model_path=dnn_model_path,
                )
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Run failed", str(e)))
                self.root.after(0, lambda: self.status.set("failed"))
            else:
                if self._stop_event.is_set():
                    self.root.after(0, lambda: self.status.set("cancelled"))
                else:
                    self.root.after(0, lambda: self.status.set("done (check outputs folder)"))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel(self) -> None:
        if not self._stop_event.is_set():
            self._stop_event.set()
            self.status.set("cancelling...")


def main() -> int:
    root = Tk()
    _ = App(root)
    root.mainloop()
    return 0

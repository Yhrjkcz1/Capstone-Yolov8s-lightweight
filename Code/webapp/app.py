# -*- coding: utf-8 -*-
"""
Gradio demo: YOLOv8s-GEA (VisDrone), GitHub-documentation-style UI.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import traceback
from typing import Any

os.environ.setdefault("DISPLAY", ":0")

import cv2
import gradio as gr
import numpy as np

from inference import Detector, detections_to_yolo_txt, safe_imwrite

# --- logging (INFO lines for model stats) ---
logger = logging.getLogger("yolov8_gea_webapp")


def _setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


# Populated after load from ultralytics helpers (also used when model fails to load)
REAL_PARAMS_DISPLAY = "—"
REAL_GFLOPS_DISPLAY = "N/A"

# ===================== Paths ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "assets", "custom.css")
WEIGHTS_PATH = r"D:\Capstone\runs\detect\yolov8s_GEA(KD)\weights\best.pt"
SAMPLES_DIR = os.path.join(BASE_DIR, "assets", "samples")

VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ===================== Global model ================================================
DETECTOR: Detector | None = None
MODEL_LOAD_ERROR: str | None = None


def _load_css() -> str:
    try:
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _init_model() -> None:
    global DETECTOR, MODEL_LOAD_ERROR
    try:
        DETECTOR = Detector(WEIGHTS_PATH, device=None)
        MODEL_LOAD_ERROR = None
    except Exception as e:
        DETECTOR = None
        MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"


def _likely_headless_gui() -> bool:
    """Heuristic: Linux without DISPLAY → tkinter dialog usually unavailable."""
    if os.name == "nt":
        return False
    if sys.platform == "darwin":
        return False
    return not bool(os.environ.get("DISPLAY", "").strip())


def _tk_pick_folder() -> str | None:
    """Open OS folder dialog; returns path or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        folder = filedialog.askdirectory()
    finally:
        root.destroy()
    return folder.strip() if folder else None


def browse_folder_action() -> str:
    """For Gradio: returns new folder path or keeps previous via caller."""
    if _likely_headless_gui():
        logger.warning(
            "Folder dialog skipped: no DISPLAY or headless environment; use manual path or file upload."
        )
        raise gr.Warning(
            "System dialog not available in this environment, "
            "please type the path manually or use file upload."
        )
    try:
        picked = _tk_pick_folder()
        return picked if picked else gr.update()
    except Exception as ex:
        logger.warning("tkinter filedialog failed (%s: %s); use manual path or file upload.", type(ex).__name__, ex)
        raise gr.Warning(
            "System dialog not available in this environment, "
            "please type the path manually or use file upload."
        ) from ex


def collect_real_model_stats(det: Detector) -> tuple[str, str]:
    """
    Read parameters and GFLOPs; log REAL MODEL METRICS block; return (params_str, flops_str) for UI.
    """
    global REAL_PARAMS_DISPLAY, REAL_GFLOPS_DISPLAY

    from ultralytics.utils.torch_utils import get_flops, get_num_params

    m = det.model.model
    n_params = int(get_num_params(m))

    flops_display = "N/A"
    try:
        raw = get_flops(m, imgsz=1024)
        if isinstance(raw, (int, float)):
            if np.isfinite(float(raw)):
                flops_display = f"{float(raw):.1f}"
            else:
                flops_display = "N/A"
        elif raw is None:
            flops_display = "N/A"
        else:
            s = str(raw).strip()
            flops_display = s if s else "N/A"
    except Exception as ex:
        logger.warning("get_flops failed (%s: %s); GFLOPs set to N/A.", type(ex).__name__, ex)
        flops_display = "N/A"

    logger.info("========== REAL MODEL METRICS ==========")
    logger.info(
        "[INFO] Real model parameters: %s (%.1f M)",
        f"{n_params:,}",
        n_params / 1e6,
    )
    logger.info("[INFO] Real model GFLOPs: %s", flops_display)
    logger.info("========================================")

    REAL_PARAMS_DISPLAY = f"{n_params / 1e6:.2f} M"
    REAL_GFLOPS_DISPLAY = flops_display

    return REAL_PARAMS_DISPLAY, REAL_GFLOPS_DISPLAY


def build_header_html() -> str:
    return """
<div class="app-header">
  <div class="header-row">
    <div>
      <h1 class="app-title">YOLOv8s-GEA · Lightweight UAV Object Detection</h1>
      <p class="app-subtitle">Knowledge Distillation Enhanced YOLOv8s for VisDrone</p>
    </div>
    <div class="badges-wrap">
      <div class="gh-badge"><span class="b-key">Model</span><span class="b-val">YOLOv8s-GEA</span></div>
      <div class="gh-badge"><span class="b-key">Dataset</span><span class="b-val">VisDrone</span></div>
      <div class="gh-badge"><span class="b-key">Framework</span><span class="b-val">Ultralytics</span></div>
    </div>
  </div>
</div>
"""


def build_metrics_html(params_str: str, flops_str: str) -> str:
    return f"""
<div class="metrics-wrap">
<div class="metrics-row">
  <div class="metric-card">
    <div class="metric-label">Parameters</div>
    <div class="metric-value">{params_str}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">GFLOPs</div>
    <div class="metric-value">{flops_str}</div>
  </div>
</div>
</div>
"""


def build_about_markdown() -> str:
    classes_txt = ", ".join(VISDRONE_CLASSES)
    return f"""
<div class="about-block">

## About

This demo presents **YOLOv8s with GEA** and **knowledge distillation (KD)** for **UAV / VisDrone-style** object detection. Interactive defaults follow common evaluation settings: `imgsz=1024`, `conf=0.25`, `iou=0.45`.

### Highlights

- Lightweight detection head tuned for **small objects** and **crowded aerial** scenes.
- KD-assisted training for a better **accuracy–efficiency** trade-off.
- **{len(VISDRONE_CLASSES)}** VisDrone-style classes: {classes_txt}.
- Use **Single Image** for quick inspection, or **Batch Folder** to export **annotated images**, **YOLO txt labels**, and a consolidated **`detections.json`** under `images/`, `labels/`, and the run root.

</div>
"""


def collect_sample_images() -> list[list[str]]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(SAMPLES_DIR, pat)))
    paths = sorted(set(paths))
    return [[p] for p in paths]


def collect_image_paths(root: str, recursive: bool) -> list[str]:
    root = os.path.abspath(os.path.expanduser(root.strip()))
    out: list[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTENSIONS:
                    out.append(os.path.join(dirpath, fn))
    else:
        if not os.path.isdir(root):
            return []
        for fn in sorted(os.listdir(root)):
            p = os.path.join(root, fn)
            if os.path.isfile(p):
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTENSIONS:
                    out.append(p)
    return sorted(out)


def _normalize_uploaded_files(uploaded_files: Any) -> list[str]:
    """Gradio File may return None, a path, tempfile-like objects, or a list of those."""

    def _one(p: Any) -> str | None:
        if p is None:
            return None
        if isinstance(p, str) and p.strip():
            return p
        name = getattr(p, "name", None)
        if isinstance(name, str) and name.strip():
            return name
        return None

    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (list, tuple)):
        out: list[str] = []
        for p in uploaded_files:
            u = _one(p)
            if u:
                out.append(u)
        return out
    u = _one(uploaded_files)
    return [u] if u else []


def run_detection(
    image_path: str | None,
    conf: float,
    iou: float,
    imgsz_choice: str,
) -> tuple[Any, str, list[list], dict[str, Any]]:
    if MODEL_LOAD_ERROR is not None:
        md = f"**Model load failed:** `{MODEL_LOAD_ERROR}`"
        return None, md, [], {"error": MODEL_LOAD_ERROR, "weights": WEIGHTS_PATH}

    if DETECTOR is None:
        md = "**Internal error:** detector is not initialized."
        return None, md, [], {"error": "detector_none"}

    if image_path is None or str(image_path).strip() == "":
        return (
            None,
            "**Hint:** Upload an image or pick an example below, then click **Run Detection**.",
            [],
            {"error": "no_image"},
        )

    imgsz = int(imgsz_choice)
    try:
        assert DETECTOR is not None
        out = DETECTOR.predict(image_path, conf=float(conf), iou=float(iou), imgsz=imgsz)
    except Exception as e:
        tb = traceback.format_exc()
        md = f"**Inference failed:** `{type(e).__name__}: {e}`\n\n```\n{tb}\n```"
        return None, md, [], {"error": str(e), "traceback": tb}

    ann_bgr = out["annotated_image"]
    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

    inf_ms = float(out["inference_ms"])
    fps_inst = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
    spd = out["speed"]
    nobj = len(out["detections"])

    md = f"""
**Detected Objects**: {nobj}

**Inference Time**: {inf_ms:.2f} ms (this run, wall-clock)

**FPS (this run, estimated)**: {fps_inst:.2f}

**Ultralytics speed**: preprocess {spd["preprocess"]:.2f} ms · inference {spd["inference"]:.2f} ms · postprocess {spd["postprocess"]:.2f} ms
""".strip()

    rows: list[list] = []
    for i, d in enumerate(out["detections"], start=1):
        bb = d["bbox"]
        bbox_str = f"({bb[0]:.1f}, {bb[1]:.1f}, {bb[2]:.1f}, {bb[3]:.1f})"
        rows.append(
            [
                i,
                d["class_name"],
                round(float(d["confidence"]), 4),
                bbox_str,
            ]
        )

    payload = {
        "weights": WEIGHTS_PATH,
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": imgsz,
        "detections": out["detections"],
        "inference_ms": out["inference_ms"],
        "speed": out["speed"],
    }

    return ann_rgb, md, rows, payload


def run_batch_detection(
    uploaded_files: Any,
    in_folder: str,
    out_folder: str,
    conf: float,
    iou: float,
    imgsz_choice: str,
    recursive: bool,
    save_ann: bool,
    save_yolo: bool,
    save_det_json: bool,
    progress=gr.Progress(),
):
    """
    Generator: live progress markdown, final summary markdown, gallery paths (first 12 annotated).
    Priority: uploaded_files > Input Folder path.
    """
    if MODEL_LOAD_ERROR is not None:
        yield (
            f"**Model load failed:** `{MODEL_LOAD_ERROR}`",
            "",
            [],
        )
        return

    if DETECTOR is None:
        yield "**Internal error:** detector is not initialized.", "", []
        return

    uploads = _normalize_uploaded_files(uploaded_files)
    in_folder = (in_folder or "").strip()
    out_folder = (out_folder or "").strip()

    use_upload = len(uploads) > 0
    if use_upload:
        paths = sorted(uploads)
        input_desc = "(multiple uploaded images)"
        base_label_path = ""  # no common root for relpath
    else:
        if not in_folder:
            yield (
                "**Error:** Please set **Input Folder** or **upload images** (batch upload takes priority).",
                "",
                [],
            )
            return
        if not os.path.isdir(in_folder):
            yield (
                "**Error:** Input folder does not exist: `{0}`".format(in_folder),
                "",
                [],
            )
            return
        paths = collect_image_paths(in_folder, recursive)
        input_desc = os.path.abspath(in_folder)
        base_label_path = os.path.abspath(in_folder)

    if not out_folder:
        yield "**Error:** Output folder path is empty.", "", []
        return

    try:
        os.makedirs(out_folder, exist_ok=True)
        images_dir = os.path.join(out_folder, "images")
        labels_dir = os.path.join(out_folder, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
    except Exception as e:
        yield f"**Error:** cannot create output folders: `{type(e).__name__}: {e}`", "", []
        return

    imgsz = int(imgsz_choice)
    total = len(paths)

    if total == 0:
        yield "**Error:** No images found (supported: jpg, jpeg, png, bmp, webp).", "", []
        return

    live_lines: list[str] = []
    failed: list[dict[str, str]] = []
    per_image: list[dict[str, Any]] = []
    total_det = 0
    ok_count = 0
    sum_ms_ok = 0.0

    det_handle = DETECTOR

    for idx, path in enumerate(paths):
        base = os.path.basename(path)
        if use_upload:
            rel = base
        else:
            rel = os.path.relpath(path, base_label_path)
        try:
            assert det_handle is not None
            out = det_handle.predict(path, conf=float(conf), iou=float(iou), imgsz=imgsz)
            ms = float(out["inference_ms"])
            dets = out["detections"]
            n_d = len(dets)
            total_det += n_d
            ok_count += 1
            sum_ms_ok += ms

            stem, ext = os.path.splitext(base)
            ext = ext if ext.lower() in IMG_EXTENSIONS else ".jpg"
            out_img_path = os.path.join(images_dir, stem + ext)
            out_lbl_path = os.path.join(labels_dir, stem + ".txt")

            ann_bgr = out["annotated_image"]
            h, w = ann_bgr.shape[:2]

            if save_ann:
                if not safe_imwrite(out_img_path, ann_bgr):
                    raise RuntimeError(f"failed to write annotated image: {out_img_path}")

            if save_yolo:
                txt_body = detections_to_yolo_txt(dets, w, h)
                try:
                    with open(out_lbl_path, "w", encoding="utf-8") as lf:
                        lf.write(txt_body)
                except Exception as le:
                    raise RuntimeError(f"failed to write label: {out_lbl_path} ({le})") from le

            per_image.append(
                {
                    "source_path": path,
                    "relative_path": rel,
                    "inference_ms": ms,
                    "num_detections": n_d,
                    "detections": dets,
                }
            )

            live_lines.append(f"Processing {idx + 1}/{total}: `{rel}` ... done ({ms:.1f} ms)")
        except Exception as e:
            failed.append({"path": path, "error": f"{type(e).__name__}: {e}"})
            live_lines.append(
                f"Processing {idx + 1}/{total}: `{rel}` ... skipped ({type(e).__name__}: {e})"
            )

        tail = live_lines[-25:]
        yield "\n\n".join(tail), "", []

        try:
            progress(
                (idx + 1) / max(total, 1),
                desc=f"Processing {idx + 1}/{total}",
            )
        except Exception:
            pass

    json_path = os.path.join(out_folder, "detections.json")
    summary_payload: dict[str, Any] = {
        "weights": WEIGHTS_PATH,
        "input_mode": "upload" if use_upload else "folder",
        "input_folder": input_desc if not use_upload else None,
        "uploaded_count": len(uploads) if use_upload else 0,
        "output_folder": os.path.abspath(out_folder),
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": imgsz,
        "recursive": bool(recursive) if not use_upload else False,
        "total_images_scanned": total,
        "processed_ok": ok_count,
        "failed_count": len(failed),
        "total_detections": total_det,
        "per_image": per_image,
        "failed": failed,
    }

    if save_det_json:
        try:
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(summary_payload, jf, ensure_ascii=False, indent=2)
        except Exception as je:
            failed.append({"path": json_path, "error": f"detections.json write failed: {je}"})

    avg_det = (total_det / ok_count) if ok_count > 0 else 0.0
    throughput_fps = (ok_count / (sum_ms_ok / 1000.0)) if sum_ms_ok > 0 else 0.0

    fail_lines = "\n".join(f"- `{f['path']}`: {f['error']}" for f in failed) if failed else "_None_"

    src_line = (
        f"- **Input source**: uploaded files ({len(uploads)} images)"
        if use_upload
        else f"- **Input folder**: `{input_desc}`"
    )

    summary_md = f"""
### Batch finished

{src_line}
- **Total images scanned**: {total}
- **Processed OK**: {ok_count}
- **Total detections**: {total_det}
- **Average detections / image (OK only)**: {avg_det:.2f}
- **Average throughput FPS (OK only, wall-clock per image)**: {throughput_fps:.2f}
- **Output directory**: `{os.path.abspath(out_folder)}`
- **Annotated images**: `{images_dir}` (enabled: {save_ann})
- **YOLO labels**: `{labels_dir}` (enabled: {save_yolo})
- **`detections.json`**: `{json_path}` (enabled: {save_det_json})

#### Skipped / failed images

{fail_lines}
""".strip()

    preview: list[str] = []
    if save_ann and os.path.isdir(images_dir):
        for fn in sorted(os.listdir(images_dir)):
            p = os.path.join(images_dir, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in IMG_EXTENSIONS:
                preview.append(p)
            if len(preview) >= 12:
                break

    final_live = "\n\n".join(live_lines[-40:])
    yield final_live, summary_md, preview


def make_app() -> gr.Blocks:
    css_text = _load_css()

    init_load_banner = ""
    if MODEL_LOAD_ERROR:
        init_load_banner = (
            f'<div class="load-error"><strong>Model load failed</strong><br/>{MODEL_LOAD_ERROR}<br/>'
            f'Check that weights exist at <code>{WEIGHTS_PATH}</code></div>'
        )

    params_card = REAL_PARAMS_DISPLAY
    flops_card = REAL_GFLOPS_DISPLAY

    if DETECTOR is not None:
        params_card, flops_card = collect_real_model_stats(DETECTOR)

    metrics_html = build_metrics_html(params_card, flops_card)

    with gr.Blocks(
        theme=gr.themes.Default(),
        css=css_text,
        title="YOLOv8s-GEA · VisDrone Demo",
    ) as demo:
        gr.HTML(init_load_banner)
        gr.HTML(build_header_html())
        gr.HTML(metrics_html)

        gr.Markdown("### Detection Console")

        with gr.Tabs():
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown('<p class="section-title">Input</p>')
                        inp_image = gr.Image(
                            type="filepath",
                            label="Upload Image",
                            height=320,
                        )
                        conf_sl = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.25,
                            step=0.05,
                            label="conf",
                        )
                        iou_sl = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.45,
                            step=0.05,
                            label="iou",
                        )
                        imgsz_radio = gr.Radio(
                            choices=["640", "1024"],
                            value="1024",
                            label="Inference Size (imgsz)",
                        )
                        run_btn = gr.Button("Run Detection", variant="primary")

                        example_rows = collect_sample_images()
                        if example_rows:
                            gr.Examples(
                                examples=example_rows,
                                inputs=[inp_image],
                                label="Examples (assets/samples)",
                            )

                    with gr.Column(scale=1):
                        gr.Markdown('<p class="section-title">Output</p>')
                        out_image = gr.Image(type="numpy", label="Detection Result")
                        out_stats = gr.Markdown(label="Statistics")
                        out_table = gr.Dataframe(
                            headers=["#", "Class", "Confidence", "bbox(x1,y1,x2,y2)"],
                            datatype=["number", "str", "number", "str"],
                            label="Detection List",
                            interactive=False,
                            wrap=True,
                        )
                        with gr.Accordion("Raw Detections (JSON)", open=False):
                            out_json = gr.JSON(label="raw")

                run_btn.click(
                    fn=run_detection,
                    inputs=[inp_image, conf_sl, iou_sl, imgsz_radio],
                    outputs=[out_image, out_stats, out_table, out_json],
                )

            with gr.Tab("Batch Folder"):
                gr.Markdown(
                    "Export layout: `output/images/` (annotated), `output/labels/` (YOLO txt), "
                    "`output/detections.json` (summary)."
                )
                with gr.Row():
                    batch_in = gr.Textbox(
                        label="Input Folder",
                        placeholder=r"D:\Capstone\Code\test\external_test",
                        scale=5,
                    )
                    btn_browse_in = gr.Button("📁 Browse (local)", scale=1)
                batch_upload = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Or select multiple images (Ctrl+A to select all in folder)",
                )
                with gr.Row():
                    batch_out = gr.Textbox(
                        label="Output Folder",
                        placeholder=r"D:\Capstone\Code\test\runs\webapp_batch",
                        scale=5,
                    )
                    btn_browse_out = gr.Button("📁 Browse (local)", scale=1)
                b_conf = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="conf")
                b_iou = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="iou")
                b_imgsz = gr.Radio(
                    choices=["640", "1024"],
                    value="1024",
                    label="Inference Size (imgsz)",
                )
                b_recursive = gr.Checkbox(label="Recursive (include subfolders)", value=False)
                b_save_img = gr.Checkbox(label="Save annotated images", value=True)
                b_save_yolo = gr.Checkbox(label="Save YOLO format labels", value=True)
                b_save_json = gr.Checkbox(label="Save detections JSON", value=True)
                batch_btn = gr.Button("Run Batch Detection", variant="primary")

                batch_live = gr.Markdown(label="Progress")
                batch_summary = gr.Markdown(label="Summary")
                batch_gallery = gr.Gallery(
                    label="Preview (first 12 annotated images)",
                    columns=4,
                    height=400,
                )

                btn_browse_in.click(fn=browse_folder_action, inputs=[], outputs=[batch_in])
                btn_browse_out.click(fn=browse_folder_action, inputs=[], outputs=[batch_out])

                batch_btn.click(
                    fn=run_batch_detection,
                    inputs=[
                        batch_upload,
                        batch_in,
                        batch_out,
                        b_conf,
                        b_iou,
                        b_imgsz,
                        b_recursive,
                        b_save_img,
                        b_save_yolo,
                        b_save_json,
                    ],
                    outputs=[batch_live, batch_summary, batch_gallery],
                )

        gr.Markdown(build_about_markdown())

    return demo


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description="YOLOv8s-GEA Gradio Demo")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share link (share=True)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port (default 7860)",
    )
    args = parser.parse_args()

    _init_model()

    demo = make_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(args.port),
        share=bool(args.share),
    )


if __name__ == "__main__":
    main()

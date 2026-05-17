# -*- coding: utf-8 -*-
"""
推理逻辑封装：YOLOv8 (Ultralytics) + 中文路径兼容读写。
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def safe_imread(path: str) -> np.ndarray | None:
    """
    读取图像（兼容中文/特殊字符路径）。
    使用 numpy 读入字节再用 cv2.imdecode，避免 cv2.imread 在 Windows 中文路径下失败。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def safe_imwrite(path: str, img: np.ndarray) -> bool:
    """
    保存图像（兼容中文路径）。失败返回 False，不抛异常。
    """
    try:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("jpg", "jpeg"):
            ext = "jpg"
        ok, buf = cv2.imencode("." + ext, img)
        if ok and buf is not None:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return False


def _resolve_device(device: str | None) -> str:
    if device is not None and device.strip():
        return device.strip()
    return "0" if torch.cuda.is_available() else "cpu"


class Detector:
    """封装 YOLO 推理，输出统一字典结构。"""

    def __init__(self, weights_path: str, device: str | None = None) -> None:
        self.weights_path = weights_path
        self.device = _resolve_device(device)
        self.model = YOLO(weights_path)

    def predict(
        self,
        image: str | np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> dict[str, Any]:
        """
        对单张图像推理。

        image: 文件路径(str) 或 BGR 图像 ndarray

        返回:
            annotated_image: BGR 标注图
            detections: 检测列表
            inference_ms: 本次 predict  wall-clock 耗时（毫秒）
            speed: Ultralytics 报告的 preprocess / inference / postprocess（毫秒）
        """
        if isinstance(image, str):
            img_bgr = safe_imread(image)
            if img_bgr is None:
                raise ValueError(f"Cannot read image file: {image}")
        else:
            img_bgr = image
            if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
                raise ValueError("Invalid image data (empty or wrong type)")

        t0 = time.perf_counter()
        results = self.model.predict(
            source=img_bgr,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=self.device,
            verbose=False,
        )
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        if not results:
            empty = np.ascontiguousarray(img_bgr)
            return {
                "annotated_image": empty,
                "detections": [],
                "inference_ms": float(inference_ms),
                "speed": {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0},
            }

        r0 = results[0]
        annotated = r0.plot()
        if annotated is None:
            annotated = np.ascontiguousarray(img_bgr)
        else:
            annotated = np.ascontiguousarray(annotated)

        # Ultralytics speed 字典（毫秒）；若不存在则填 0
        spd = getattr(r0, "speed", None) or {}
        speed_out = {
            "preprocess": float(spd.get("preprocess", 0.0) or 0.0),
            "inference": float(spd.get("inference", 0.0) or 0.0),
            "postprocess": float(spd.get("postprocess", 0.0) or 0.0),
        }

        names = self.model.names if hasattr(self.model, "names") else {}
        detections: list[dict[str, Any]] = []

        boxes = getattr(r0, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                cid = int(cls_arr[i])
                cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
                detections.append(
                    {
                        "class_id": cid,
                        "class_name": str(cname),
                        "confidence": float(conf_arr[i]),
                        "bbox": [
                            float(xyxy[i][0]),
                            float(xyxy[i][1]),
                            float(xyxy[i][2]),
                            float(xyxy[i][3]),
                        ],
                    }
                )

        return {
            "annotated_image": annotated,
            "detections": detections,
            "inference_ms": float(inference_ms),
            "speed": speed_out,
        }


def xyxy_to_yolo_line(class_id: int, bbox_xyxy: list[float], img_w: int, img_h: int) -> str:
    """
    Convert one xyxy box (pixel coords) to one YOLO line: cls xc yc w h (normalized).
    """
    x1, y1, x2, y2 = bbox_xyxy
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Invalid image size for YOLO export")
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    return f"{int(class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def detections_to_yolo_txt(detections: list[dict[str, Any]], img_w: int, img_h: int) -> str:
    """Build YOLO-format label file content from Detector predictions."""
    lines: list[str] = []
    for d in detections:
        lines.append(
            xyxy_to_yolo_line(int(d["class_id"]), list(d["bbox"]), img_w, img_h)
        )
    return "\n".join(lines) + ("\n" if lines else "")

# -*- coding: utf-8 -*-
"""
YOLO视觉检测工具
封装YOLOv8模型加载与推理，返回结构化检测结果
"""
import os
import torch
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, CONF_THRESHOLD, HAZARD_CLASS_MAP, LOG_ENABLED

_model = None


def _get_model():
    global _model
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if LOG_ENABLED:
            print("[YOLO] 加载模型: %s, 设备: %s" % (YOLO_MODEL_PATH, device))
        _model = YOLO(YOLO_MODEL_PATH).to(device)
    return _model


def _log(msg: str):
    if LOG_ENABLED:
        print("[YOLO] " + msg)


def yolo_detect(img_path: str, conf_thresh: float = None) -> dict:
    """
    YOLO目标检测
    :param img_path: 图片路径
    :param conf_thresh: 置信度阈值，默认使用配置值
    :return: {
        "success": bool, "detections": [...], "error": str,
        "img_path": str, "img_name": str
    }
    detection结构: {
        "class_name": str,     # YOLO原始类别名，如 person, suitcase, bottle
        "hazard_type": str,    # 映射后的隐患类型，如 area_intrusion, channel_stack
        "confidence": float,
        "bbox": [x1,y1,x2,y2]
    }
    """
    if conf_thresh is None:
        conf_thresh = 0.3  # 返回较多结果，由LLM/台账判断是否为隐患，而不是YOLO阈值卡死

    result = {
        "success": False,
        "detections": [],
        "error": "",
        "img_path": img_path,
        "img_name": os.path.basename(img_path)
    }

    if not os.path.exists(img_path):
        result["error"] = "图片不存在: %s" % img_path
        _log(result["error"])
        return result

    try:
        model = _get_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _log("检测: %s, conf>=%.2f" % (os.path.basename(img_path), conf_thresh))

        outputs = model.predict(img_path, device=device, conf=conf_thresh)

        detections = []
        for res in outputs:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cls_name = res.names[cls_id]
                hazard_type = HAZARD_CLASS_MAP.get(cls_name, "other")

                detections.append({
                    "class_name": cls_name,
                    "hazard_type": hazard_type,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in [x1, y1, x2, y2]]
                })

        result["success"] = True
        result["detections"] = detections
        _log("完成: %d 个目标" % len(detections))
        return result

    except Exception as e:
        result["error"] = "YOLO检测异常: %s: %s" % (type(e).__name__, str(e))
        _log(result["error"])
        return result

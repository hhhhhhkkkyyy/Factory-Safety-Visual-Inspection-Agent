# -*- coding: utf-8 -*-
"""
检测结果可视化工具 - 创新功能
将YOLO检测结果绘制到图片上，生成带标注的证据图
支持：不同隐患类型用不同颜色标注、置信度显示、隐患类型标签
"""
import os
import cv2
import numpy as np
from config import LOG_ENABLED


# 隐患类型 -> 颜色 (BGR格式)
HAZARD_COLORS = {
    "no_helmet": (0, 0, 255),       # 红色 - 高危
    "area_intrusion": (0, 0, 255),  # 红色 - 高危
    "channel_stack": (0, 165, 255), # 橙色 - 一般
    "person_detected": (255, 0, 0), # 蓝色 - 待确认
    "other": (128, 128, 128),       # 灰色 - 其他
}


def _log(msg: str):
    if LOG_ENABLED:
        print(f"[VIS] {msg}")


def draw_detections(img_path: str, detections: list, output_path: str = None) -> str:
    """
    在图片上绘制检测框和标签
    :param img_path: 原图路径
    :param detections: 检测结果列表
    :param output_path: 输出路径，默认在原图同目录生成 _annotated 后缀
    :return: 生成的标注图片路径
    """
    if not os.path.exists(img_path):
        _log(f"图片不存在: {img_path}")
        return ""

    try:
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            _log(f"无法读取图片: {img_path}")
            return ""

        # 绘制检测框
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)
            hazard_type = det.get("hazard_type", "other")
            conf = det.get("confidence", 0)
            class_name = det.get("class_name", "")

            color = HAZARD_COLORS.get(hazard_type, (128, 128, 128))

            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 标签背景
            label = f"{hazard_type} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)

            # 标签文字
            cv2.putText(img, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 生成输出路径
        if output_path is None:
            base, ext = os.path.splitext(img_path)
            output_path = f"{base}_annotated{ext}"

        cv2.imwrite(output_path, img)
        _log(f"标注图已生成: {output_path}")
        return output_path

    except Exception as e:
        _log(f"可视化失败: {type(e).__name__}: {e}")
        return ""


def generate_summary_image(img_paths: list, all_detections: list, output_path: str) -> str:
    """
    生成汇总对比图（创新功能：多图拼接展示）
    :param img_paths: 多张图片路径
    :param all_detections: 对应每张图的检测结果
    :param output_path: 输出路径
    :return: 汇总图路径
    """
    annotated_imgs = []
    for img_path, dets in zip(img_paths, all_detections):
        ann_path = draw_detections(img_path, dets)
        if ann_path and os.path.exists(ann_path):
            img = cv2.imread(ann_path)
            if img is not None:
                # 统一宽度
                target_w = 640
                h, w = img.shape[:2]
                scale = target_w / w
                img = cv2.resize(img, (target_w, int(h * scale)))
                annotated_imgs.append(img)

    if not annotated_imgs:
        return ""

    # 横向拼接
    try:
        # 取最大高度
        max_h = max(img.shape[0] for img in annotated_imgs)
        # 统一高度
        resized = []
        for img in annotated_imgs:
            if img.shape[0] < max_h:
                pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            resized.append(img)

        summary = np.hstack(resized)
        cv2.imwrite(output_path, summary)
        _log(f"汇总图已生成: {output_path}")
        return output_path
    except Exception as e:
        _log(f"生成汇总图失败: {e}")
        return ""

# -*- coding: utf-8 -*-
"""
离线模拟LLM客户端
注意：Mock是固定规则的if-else，不会真正理解自然语言指令
用于开发调试、无网环境、性能测试
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONF_THRESHOLD, DEFAULT_RISK_LEVEL, LOG_ENABLED


def _log(msg: str):
    if LOG_ENABLED:
        print("[MOCK_LLM] " + msg)


class MockArkClient:
    """模拟豆包API（固定规则，用于离线/演示）"""

    def __init__(self):
        self.consecutive_failures = 0
        self.max_consecutive_failures = 2

    def plan_action(self, state: dict) -> dict:
        """模拟LLM决策（固定规则，不理解prompt）"""
        round_count = state.get("round_count", 0)
        detect_results = state.get("detect_results", [])
        ledger_info = state.get("ledger_info")
        img_list = state.get("img_list", [])
        current_idx = state.get("current_img_index", -1)

        _log("第%d轮模拟决策 | 已检测%d张 | 台账:%s" % (
            round_count, len(detect_results),
            "已查" if ledger_info else "未查"))

        # 规则1：还没检测 -> 先检测第一张
        if len(detect_results) == 0:
            return {
                "action": "yolo_detect",
                "action_args": {"img_index": 0},
                "thought": "尚未获取检测结果，先调用YOLO检测第一张图片",
                "report": "",
                "need_human_review": False
            }

        # 获取最后一次检测
        last_det = detect_results[-1]
        detections = last_det.get("detections", [])

        # 找出在阈值附近（需复核区间）的目标
        review_low = CONF_THRESHOLD - 0.1
        review_high = CONF_THRESHOLD

        # 获取台账的违禁清单（如果已查）
        forbidden = None
        hazard_map = None
        if ledger_info and ledger_info.get("success"):
            forbidden = ledger_info.get("forbidden_objects", [])
            hazard_map = ledger_info.get("hazard_mapping", {})

        # 筛选可疑/待复检目标
        review_candidates = []
        for d in detections:
            cls_name = d.get("class_name", "")
            conf = d.get("confidence", 0)
            # 在复核区间的目标
            if review_low <= conf <= review_high:
                # 如果有违禁清单，只看违禁的
                if forbidden is None or cls_name in forbidden:
                    review_candidates.append(d)

        # 规则2：有复检候选，且还有未检测的图 -> 复检
        if review_candidates and current_idx + 1 < len(img_list):
            next_idx = current_idx + 1
            return {
                "action": "yolo_detect",
                "action_args": {"img_index": next_idx},
                "thought": "检测到%d个低置信可疑目标，使用第%d张图复检确认" % (
                    len(review_candidates), next_idx + 1),
                "report": "",
                "need_human_review": False
            }

        # 规则3：有检测结果但没查台账 -> 查台账
        if ledger_info is None:
            return {
                "action": "query_workshop_ledger",
                "action_args": {"area_name": state.get("area", "")},
                "thought": "已有检测结果，需查询台账获取违禁清单、责任人与风险等级",
                "report": "",
                "need_human_review": False
            }

        # 规则4：信息齐全 -> 生成报告finish
        report, need_review = self._generate_report(state, detect_results, ledger_info)
        return {
            "action": "finish",
            "action_args": {},
            "thought": "检测结果和台账信息均已齐全，生成最终巡检报告",
            "report": report,
            "need_human_review": need_review
        }

    def too_many_failures(self) -> bool:
        return self.consecutive_failures >= self.max_consecutive_failures

    def _generate_report(self, state: dict, detect_results: list, ledger_info: dict):
        """模拟生成报告（基于台账规则判定）"""
        area = state.get("area", "")
        round_count = state.get("round_count", 0)

        # 获取台账规则
        forbidden = []
        hazard_map = {}
        person = "【台账信息缺失】"
        contact = ""
        risk_base = "未知"
        safety_rule = ""
        ledger_ok = False

        if ledger_info and ledger_info.get("success"):
            ledger_ok = True
            forbidden = ledger_info.get("forbidden_objects", [])
            hazard_map = ledger_info.get("hazard_mapping", {})
            person = ledger_info.get("person_in_charge", "未知")
            contact = ledger_info.get("contact", "")
            risk_base = ledger_info.get("risk_base_level", "未知")
            safety_rule = ledger_info.get("safety_rule", "")

        # 汇总隐患
        hazards = []
        review_items = []
        review_low = CONF_THRESHOLD - 0.1

        for det in detect_results:
            img_name = det.get("img_name", "")
            for d in det.get("detections", []):
                cls_name = d.get("class_name", "")
                conf = d.get("confidence", 0)

                # 用台账规则判定
                if hazard_map and cls_name in hazard_map:
                    hinfo = hazard_map[cls_name]
                    htype = hinfo.get("type", "other")
                    hlevel = hinfo.get("level", "待确认")
                    hdesc = hinfo.get("desc", cls_name)

                    # 正常物品不算隐患
                    if htype in ["normal_item", "normal_facility"] or hlevel == "无":
                        continue

                    hazards.append({
                        "type": htype,
                        "class_name": cls_name,
                        "conf": conf,
                        "img": img_name,
                        "risk": hlevel,
                        "desc": hdesc
                    })

                    # 检查是否在复核区间
                    if review_low <= conf <= CONF_THRESHOLD:
                        review_items.append(hazards[-1])

                elif forbidden and cls_name in forbidden:
                    # 在违禁清单但没有详细映射
                    hlevel = DEFAULT_RISK_LEVEL.get("channel_stack", "一般隐患")
                    hazards.append({
                        "type": "channel_stack",
                        "class_name": cls_name,
                        "conf": conf,
                        "img": img_name,
                        "risk": hlevel,
                        "desc": cls_name
                    })
                    if review_low <= conf <= CONF_THRESHOLD:
                        review_items.append(hazards[-1])
                # 不在违禁清单 -> 不算隐患，跳过

        need_review = len(review_items) > 0

        # 构建报告
        report = "# 厂区安全巡检报告\n"
        report += "巡检区域：%s\n" % area
        report += "执行轮次：%d\n" % round_count
        report += "人工复核标记：%s\n" % ("是" if need_review else "否")

        report += "\n## 隐患清单\n"
        if not hazards:
            report += "✅ **本次巡检未识别到安全隐患**\n"
        else:
            report += "| 序号 | 隐患类型 | 原始类别 | 置信度 | 证据图片 | 风险等级 | 说明 |\n"
            report += "| ---- | ---- | ---- | ---- | ---- | ---- | ---- |\n"
            for i, h in enumerate(hazards, 1):
                flag = " ⚠️待复核" if h in review_items else ""
                report += "| %d | %s | %s | %.2f | %s | %s | %s%s |\n" % (
                    i, h["type"], h["class_name"], h["conf"],
                    h["img"], h["risk"], h["desc"], flag)

        report += "\n## 区域责任人\n"
        report += "姓名：%s\n" % person
        if contact:
            report += "联系电话：%s\n" % contact
        if risk_base != "未知":
            report += "区域等级：%s\n" % risk_base

        report += "\n## 处置建议\n"
        if not ledger_ok:
            report += "【台账信息缺失】以下建议基于通用安全规则生成。\n\n"

        high_risk = [h for h in hazards if h["risk"] == "高危告警"]
        normal_risk = [h for h in hazards if h["risk"] == "一般隐患"]

        if high_risk:
            report += "⚠️ **高危告警**：检测到 %d 项高危隐患。" % len(high_risk)
            if safety_rule:
                report += safety_rule
            else:
                report += "请立即通知责任人现场整改。"
            report += "\n"
            if person and person != "【台账信息缺失】":
                report += "责任人：%s" % person
                if contact:
                    report += "，联系电话：%s" % contact
                report += "\n"

        if normal_risk:
            report += "📋 **一般隐患**：检测到 %d 项一般隐患，" % len(normal_risk)
            report += "请责任人纳入整改计划，下次巡检复核。\n"

        if not hazards:
            report += "本次巡检区域状况良好，未发现安全隐患。继续保持。\n"

        if need_review:
            report += "\n⚠️ 【待人工复核】检测到 %d 个目标置信度在 %.2f~%.2f 之间，" % (
                len(review_items), review_low, CONF_THRESHOLD)
            report += "AI判定存疑，请人工查看标注图片确认是否为隐患，暂不自动触发高危告警。\n"

        return report, need_review

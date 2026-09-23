# -*- coding: utf-8 -*-
"""
Agent状态机核心 - 轻量级LangGraph替代实现
核心思想与LangGraph一致：状态流转 + 节点调度 + 循环执行

三个核心节点：
  1. llm_planner  - LLM规划节点，决策下一步动作
  2. tool_executor - 工具执行节点，调用YOLO/台账等工具
  3. judge_stop   - 停止判断节点，决定是否终止循环

双模式：
- online  : 真实豆包API（需要网络和API Key）
- offline : 内置Mock LLM（离线可用，用于开发调试演示）

创新功能：
- 自动生成检测标注图（证据截图）
- 报告自动保存为本地Markdown文件
- 多图复检对比汇总图
- 在线/离线双模式自动降级
"""
import os
import json
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state_def import AgentState
from tools.yolo_tool import yolo_detect
from tools.ledger_tool import query_workshop_ledger
from tools.visualize_tool import draw_detections, generate_summary_image
from config import MAX_ROUND, LOG_ENABLED


def _log(msg: str):
    if LOG_ENABLED:
        print("\n" + "=" * 60)
        print("[AGENT] " + msg)
        print("=" * 60)


class InspectionAgent:
    """
    厂区安全巡检视觉Agent
    基于状态机的循环执行模型，思想与LangGraph一致
    """

    def __init__(self, output_dir: str = None, mode: str = "auto"):
        """
        :param mode: online / offline / auto
                     auto模式优先尝试在线，失败自动降级到离线
        """
        self.mode = mode
        self.llm = None
        self._init_llm()

        # 报告输出目录
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_llm(self):
        """初始化LLM客户端，支持自动降级"""
        if self.mode == "offline":
            from mock_llm import MockArkClient
            self.llm = MockArkClient()
            _log("使用离线模式（Mock LLM）")
            return

        # 尝试在线模式
        try:
            from llm_client import ArkClient
            self.llm = ArkClient()
            _log("使用在线模式（豆包API）")
        except Exception as e:
            if self.mode == "auto":
                from mock_llm import MockArkClient
                self.llm = MockArkClient()
                self.mode = "offline"
                _log("在线模式初始化失败，自动降级到离线模式: %s" % e)
            else:
                raise

    def run(self, user_prompt: str, area: str, img_list: list) -> dict:
        """
        对外入口：执行一次完整巡检
        :param user_prompt: 用户自然语言指令
        :param area: 巡检区域名
        :param img_list: 图片路径列表（第一张主图，后续用于复检）
        :return: {"report": str, "need_human_review": bool, "round_count": int,
                  "report_path": str, "annotated_images": list}
        """
        _log("启动巡检任务 | 区域: %s | 图片数: %d | 模式: %s" % (
            area, len(img_list), self.mode))
        _log("用户指令: %s" % user_prompt)

        # 初始化状态
        state: AgentState = {
            "user_prompt": user_prompt,
            "area": area,
            "img_list": img_list,
            "round_count": 0,
            "detect_results": [],
            "ledger_info": None,
            "current_img_index": -1,
            "error_log": [],
            "action": "",
            "action_args": {},
            "thought": "",
            "report": "",
            "need_human_review": False,
            "is_finished": False,
        }

        # 主循环：状态机流转
        while not state["is_finished"] and state["round_count"] < MAX_ROUND:
            # 节点1: LLM规划
            state = self._node_llm_planner(state)

            # 在线模式下，如果API失败且是auto模式，尝试降级
            if self.mode == "online" and state["action"] == "error":
                if hasattr(self.llm, 'consecutive_failures') and self.llm.consecutive_failures >= 1:
                    _log("在线API调用失败，尝试降级到离线模式")
                    from mock_llm import MockArkClient
                    self.llm = MockArkClient()
                    self.mode = "offline"
                    state["action"] = ""  # 重置动作，重新规划
                    # 不增加轮次，用本轮继续
                    state["round_count"] -= 1
                    state["error_log"].append("在线API不可用，已自动降级到离线模式")
                    continue

            # 节点2: 工具执行
            if state["action"] in ["yolo_detect", "query_workshop_ledger"]:
                state = self._node_tool_executor(state)

            # 节点3: 停止判断
            state = self._node_judge_stop(state)

        # 循环结束后的收尾处理
        if state["round_count"] >= MAX_ROUND and not state["is_finished"]:
            _log("达到最大轮次，强制终止")
            state["need_human_review"] = True
            state["report"] = self._build_force_stop_report(state)

        # ===== 创新功能：后处理 =====
        annotated_images = self._generate_evidence_images(state)
        summary_img = self._generate_summary_image(state)
        report_path = self._save_report(state, annotated_images, summary_img)

        _log("巡检完成 | 轮次: %d | 人工复核: %s" % (
            state["round_count"], state["need_human_review"]))
        _log("报告已保存: %s" % report_path)

        return {
            "report": state["report"],
            "need_human_review": state["need_human_review"],
            "round_count": state["round_count"],
            "report_path": report_path,
            "annotated_images": annotated_images,
            "summary_image": summary_img,
            "mode": self.mode,
        }

    # ========== 核心节点 ==========

    def _node_llm_planner(self, state: AgentState) -> AgentState:
        """节点1: LLM规划节点"""
        state["round_count"] += 1
        _log("[节点: llm_planner] 第%d轮规划" % state["round_count"])

        if self.llm.too_many_failures():
            _log("连续API调用失败次数超限，强制终止")
            state["action"] = "finish"
            state["thought"] = "连续API调用失败，服务异常"
            state["need_human_review"] = True
            state["report"] = self._build_error_report(
                state, "大模型API连续调用失败，服务异常")
            return state

        llm_output = self.llm.plan_action(state)

        state["action"] = llm_output.get("action", "retry")
        state["action_args"] = llm_output.get("action_args", {})
        state["thought"] = llm_output.get("thought", "")

        if state["action"] == "finish":
            state["report"] = llm_output.get("report", "")
            # 同步人工复核标记
            if "need_human_review" in llm_output:
                state["need_human_review"] = llm_output["need_human_review"]

        if LOG_ENABLED:
            print("  决策动作: %s" % state["action"])
            print("  思考: %s" % state["thought"][:100])
            print("  参数: %s" % state["action_args"])

        return state

    def _node_tool_executor(self, state: AgentState) -> AgentState:
        """节点2: 工具执行节点"""
        _log("[节点: tool_executor] 执行工具: %s" % state["action"])

        action = state["action"]
        args = state["action_args"]

        try:
            if action == "yolo_detect":
                img_index = args.get("img_index", 0)
                img_list = state["img_list"]
                if img_index >= len(img_list):
                    state["error_log"].append(
                        "图片索引%d超出范围，共有%d张" % (img_index, len(img_list)))
                    return state

                img_path = img_list[img_index]
                result = yolo_detect(img_path)

                if result["success"]:
                    state["detect_results"].append(result)
                    state["current_img_index"] = img_index
                    if LOG_ENABLED:
                        print("  检测到 %d 个目标" % len(result["detections"]))
                else:
                    state["error_log"].append(result["error"])
                    state["need_human_review"] = True
                    if LOG_ENABLED:
                        print("  检测失败: %s" % result["error"])

            elif action == "query_workshop_ledger":
                area_name = args.get("area_name", state["area"])
                result = query_workshop_ledger(area_name)
                state["ledger_info"] = result

                if LOG_ENABLED:
                    if result["success"]:
                        print("  责任人: %s" % result["person_in_charge"])
                    else:
                        print("  查询失败: %s" % result["error"])

        except Exception as e:
            error_msg = "工具执行异常: %s: %s" % (type(e).__name__, str(e))
            state["error_log"].append(error_msg)
            _log("  异常: %s" % error_msg)

        return state

    def _node_judge_stop(self, state: AgentState) -> AgentState:
        """节点3: 停止判断节点"""
        _log("[节点: judge_stop] 判断是否终止")

        # 终止条件1: action == finish
        if state["action"] == "finish":
            state["is_finished"] = True
            if LOG_ENABLED:
                print("  命中 finish，终止循环")
            return state

        # 终止条件2: 达到最大轮次
        if state["round_count"] >= MAX_ROUND:
            state["is_finished"] = True
            state["need_human_review"] = True
            if LOG_ENABLED:
                print("  达到最大轮次，强制终止")
            return state

        # 终止条件3: 连续API失败（仅在线模式）
        if self.mode == "online" and self.llm.too_many_failures():
            state["is_finished"] = True
            state["need_human_review"] = True
            state["report"] = self._build_error_report(state, "大模型API连续调用失败")
            if LOG_ENABLED:
                print("  API连续失败，强制终止")
            return state

        if LOG_ENABLED:
            print("  继续下一轮 (当前: %d/%d)" % (state["round_count"], MAX_ROUND))

        return state

    # ========== 创新功能：证据图生成与报告保存 ==========

    def _generate_evidence_images(self, state: AgentState) -> list:
        """生成带检测框的证据标注图"""
        annotated = []
        for det_result in state["detect_results"]:
            if not det_result.get("success"):
                continue
            img_path = det_result.get("img_path", "")
            detections = det_result.get("detections", [])
            if img_path and detections:
                img_name = os.path.basename(img_path)
                base, ext = os.path.splitext(img_name)
                output_path = os.path.join(
                    self.output_dir, "%s_annotated%s" % (base, ext))
                result_path = draw_detections(img_path, detections, output_path)
                if result_path:
                    annotated.append({
                        "original": img_path,
                        "annotated": result_path,
                        "img_name": img_name,
                        "detection_count": len(detections)
                    })
        return annotated

    def _generate_summary_image(self, state: AgentState) -> str:
        """生成多图汇总对比图（复检场景很有用）"""
        if len(state["detect_results"]) < 2:
            return ""

        img_paths = []
        all_dets = []
        for det_result in state["detect_results"]:
            if det_result.get("success"):
                img_paths.append(det_result["img_path"])
                all_dets.append(det_result["detections"])

        if len(img_paths) < 2:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, "summary_%s.jpg" % timestamp)
        return generate_summary_image(img_paths, all_dets, output_path)

    def _save_report(self, state: AgentState, annotated_images: list, summary_img: str) -> str:
        """保存报告为Markdown文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        area_safe = state["area"].replace("/", "_").replace("\\", "_")
        filename = "巡检报告_%s_%s.md" % (area_safe, timestamp)
        filepath = os.path.join(self.output_dir, filename)

        report_content = state["report"]

        # 附加证据图信息
        if annotated_images:
            report_content += "\n\n---\n\n## 证据图附件\n"
            for i, ann in enumerate(annotated_images, 1):
                report_content += "%d. %s（检测到%d个目标）\n" % (
                    i, ann["img_name"], ann["detection_count"])
                report_content += "   - 标注图: `%s`\n" % ann["annotated"]

        if summary_img:
            report_content += "\n汇总对比图: `%s`\n" % summary_img

        # 附加巡检元数据
        report_content += "\n---\n\n## 巡检元数据\n"
        report_content += "- 巡检时间: %s\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content += "- 用户指令: %s\n" % state["user_prompt"]
        report_content += "- 执行轮次: %d\n" % state["round_count"]
        report_content += "- 运行模式: %s\n" % self.mode
        report_content += "- 检测图片数: %d\n" % len(state["detect_results"])
        if state["error_log"]:
            report_content += "- 错误日志:\n"
            for err in state["error_log"]:
                report_content += "  - %s\n" % err

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        return filepath

    # ========== 报告模板（异常场景兜底） ==========

    def _build_error_report(self, state: AgentState, error_msg: str) -> str:
        """构建服务异常报告"""
        return """# 厂区安全巡检报告
巡检区域：%s
执行轮次：%d
人工复核标记：是

## 巡检状态
⚠️ **服务异常**：%s

## 错误日志
%s

## 处置建议
请检查系统服务状态，稍后重试；或安排人工现场巡检。""" % (
            state["area"], state["round_count"], error_msg,
            "\n".join("- " + e for e in state["error_log"])
        )

    def _build_force_stop_report(self, state: AgentState) -> str:
        """构建达到最大轮次的强制终止报告"""
        return """# 厂区安全巡检报告
巡检区域：%s
执行轮次：%d
人工复核标记：是

## 巡检状态
⚠️ **执行超时**：达到最大思考轮次%d，任务未正常完成

## 已获取信息
- 检测结果数：%d
- 台账状态：%s

## 处置建议
建议人工复核，确认巡检结果。""" % (
            state["area"], state["round_count"], MAX_ROUND,
            len(state["detect_results"]),
            "已查询" if state["ledger_info"] else "未查询"
        )

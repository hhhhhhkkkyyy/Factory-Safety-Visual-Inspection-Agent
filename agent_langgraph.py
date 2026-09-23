# -*- coding: utf-8 -*-
"""
LangGraph版本的巡检Agent
使用LangGraph构建状态机：planner -> judge_stop -> tool_executor -> planner ...

工程鲁棒性：
- planner节点检测到LLM返回 action="error" 时主动抛异常
- 异常向上传递到 main.py 的 try-except，自动降级到 offline 模式
- main.py 完全不需要修改，降级逻辑天然复用
"""
import os
import sys
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any

from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAX_ROUND, LOG_ENABLED
from tools.yolo_tool import yolo_detect
from tools.ledger_tool import query_workshop_ledger
from tools.visualize_tool import draw_detections, generate_summary_image


def _log(msg: str):
    if LOG_ENABLED:
        print("\n" + "=" * 60)
        print("[LANGGRAPH] " + msg)
        print("=" * 60)


# ========== 状态定义 ==========
class InspectionState(TypedDict, total=False):
    user_prompt: str
    area: str
    img_list: List[str]
    round_count: int
    detect_results: List[Dict[str, Any]]
    ledger_info: Optional[Dict[str, Any]]
    current_img_index: int
    error_log: List[str]
    action: str
    action_args: Dict[str, Any]
    thought: str
    report: str
    need_human_review: bool


class LangGraphAgent:
    """基于LangGraph的巡检Agent"""

    def __init__(self, llm_client, output_dir: str = None):
        self.llm = llm_client
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建LangGraph状态图"""
        workflow = StateGraph(InspectionState)

        # 三个核心节点
        workflow.add_node("planner", self._node_planner)
        workflow.add_node("tool_executor", self._node_tool_executor)
        workflow.add_node("judge_stop", self._node_judge_stop)

        # 入口
        workflow.set_entry_point("planner")

        # planner -> judge_stop
        workflow.add_edge("planner", "judge_stop")

        # judge_stop 条件路由
        workflow.add_conditional_edges(
            "judge_stop",
            self._route_after_judge,
            {
                "tool": "tool_executor",
                "finish": END,
                "retry": "planner",
            }
        )

        # tool_executor -> planner
        workflow.add_edge("tool_executor", "planner")

        return workflow.compile()

    # ========== 节点实现 ==========

    def _node_planner(self, state: InspectionState) -> dict:
        """
        规划节点：轮次+1，调用LLM
        关键：如果LLM返回 action="error"（API失败/超时），主动抛异常
              → 异常冒泡到 main.py → 自动降级 offline
        """
        current_round = state.get("round_count", 0) + 1
        _log("[节点: planner] 第%d轮规划" % current_round)

        # 用临时状态传给LLM（带上更新后的轮次）
        temp_state = dict(state)
        temp_state["round_count"] = current_round

        llm_output = self.llm.plan_action(temp_state)

        # ===== 新增：LLM调用失败时主动抛异常，触发main.py自动降级 =====
        if llm_output.get("action") == "error":
            err_msg = llm_output.get("thought", "LLM调用失败")
            _log("LLM调用失败，抛出异常触发降级: %s" % err_msg)
            raise RuntimeError("[LangGraph] LLM服务不可用: %s" % err_msg)

        updates = {
            "round_count": current_round,
            "action": llm_output.get("action", "retry"),
            "action_args": llm_output.get("action_args", {}),
            "thought": llm_output.get("thought", ""),
        }

        if updates["action"] == "finish":
            updates["report"] = llm_output.get("report", "")
            if "need_human_review" in llm_output:
                updates["need_human_review"] = llm_output["need_human_review"]

        if LOG_ENABLED:
            print("  决策动作: %s" % updates["action"])
            print("  思考: %s" % updates["thought"][:100])
            print("  参数: %s" % updates["action_args"])

        return updates

    def _node_tool_executor(self, state: InspectionState) -> dict:
        """工具执行节点"""
        action = state.get("action", "")
        args = state.get("action_args", {})
        _log("[节点: tool_executor] 执行工具: %s" % action)

        updates = {}

        try:
            if action == "yolo_detect":
                img_index = args.get("img_index", 0)
                img_list = state.get("img_list", [])

                if img_index >= len(img_list):
                    updates["error_log"] = state.get("error_log", []) + [
                        "图片索引%d超出范围" % img_index]
                    return updates

                img_path = img_list[img_index]
                result = yolo_detect(img_path)

                if result["success"]:
                    new_results = state.get("detect_results", []) + [result]
                    updates["detect_results"] = new_results
                    updates["current_img_index"] = img_index
                    if LOG_ENABLED:
                        print("  检测到 %d 个目标" % len(result["detections"]))
                else:
                    updates["error_log"] = state.get("error_log", []) + [result["error"]]
                    updates["need_human_review"] = True
                    if LOG_ENABLED:
                        print("  检测失败: %s" % result["error"])

            elif action == "query_workshop_ledger":
                area_name = args.get("area_name", state.get("area", ""))
                result = query_workshop_ledger(area_name)
                updates["ledger_info"] = result
                if LOG_ENABLED:
                    if result["success"]:
                        print("  台账查询成功 | 责任人: %s | 违禁品: %d项" % (
                            result.get("person_in_charge", "?"),
                            len(result.get("forbidden_objects", []))))
                    else:
                        print("  台账查询失败: %s" % result.get("error", ""))

            else:
                if LOG_ENABLED:
                    print("  未知动作: %s，跳过" % action)
                updates["error_log"] = state.get("error_log", []) + [
                    "未知动作: %s" % action]

        except Exception as e:
            updates["error_log"] = state.get("error_log", []) + [
                "工具执行异常: %s: %s" % (type(e).__name__, str(e))]
            if LOG_ENABLED:
                print("  工具执行异常: %s" % e)

        return updates

    def _node_judge_stop(self, state: InspectionState) -> dict:
        """判断节点：只打印日志，实际路由在 _route_after_judge"""
        action = state.get("action", "")
        _log("[节点: judge_stop] 判断是否终止 | action=%s | 轮次=%d" % (
            action, state.get("round_count", 0)))

        if LOG_ENABLED:
            if action == "finish":
                print("  生成最终报告")
            elif action in ["yolo_detect", "query_workshop_ledger"]:
                print("  执行工具后继续")
            else:
                print("  重试规划")

        return {}

    def _route_after_judge(self, state: InspectionState) -> str:
        """条件路由：tool / finish / retry"""
        action = state.get("action", "")
        round_count = state.get("round_count", 0)

        if action == "finish":
            return "finish"
        if round_count >= MAX_ROUND:
            return "finish"
        if action in ["yolo_detect", "query_workshop_ledger"]:
            return "tool"
        return "retry"

    # ========== 运行入口 ==========

    def run(self, user_prompt: str, area: str, img_list: list) -> dict:
        """执行巡检"""
        _log("启动LangGraph巡检 | 区域: %s | 图片数: %d" % (area, len(img_list)))

        initial_state: InspectionState = {
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
        }

        final_state = self.graph.invoke(initial_state)

        # 后处理
        annotated = self._gen_evidence_images(final_state)
        summary = self._gen_summary_image(final_state)
        report_path = self._save_report(final_state, annotated, summary)

        _log("巡检完成 | 轮次: %d | 人工复核: %s" % (
            final_state.get("round_count", 0),
            final_state.get("need_human_review", False)))

        return {
            "report": final_state.get("report", ""),
            "need_human_review": final_state.get("need_human_review", False),
            "round_count": final_state.get("round_count", 0),
            "report_path": report_path,
            "annotated_images": annotated,
            "summary_image": summary,
            "mode": "online_langgraph",
        }

    # ========== 后处理 ==========

    def _gen_evidence_images(self, state: dict) -> list:
        annotated = []
        for det in state.get("detect_results", []):
            if not det.get("success"):
                continue
            img_path = det.get("img_path", "")
            detections = det.get("detections", [])
            if img_path and detections:
                img_name = os.path.basename(img_path)
                base, ext = os.path.splitext(img_name)
                out = os.path.join(self.output_dir, "%s_annotated%s" % (base, ext))
                result_path = draw_detections(img_path, detections, out)
                if result_path:
                    annotated.append({
                        "original": img_path, "annotated": result_path,
                        "img_name": img_name, "detection_count": len(detections)
                    })
        return annotated

    def _gen_summary_image(self, state: dict) -> str:
        dets = state.get("detect_results", [])
        if len(dets) < 2:
            return ""
        img_paths = [d["img_path"] for d in dets if d.get("success")]
        all_dets = [d["detections"] for d in dets if d.get("success")]
        if len(img_paths) < 2:
            return ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self.output_dir, "summary_%s.jpg" % ts)
        return generate_summary_image(img_paths, all_dets, out)

    def _save_report(self, state: dict, annotated: list, summary: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        area_safe = state.get("area", "").replace("/", "_").replace("\\", "_")
        filename = "巡检报告_%s_%s.md" % (area_safe, ts)
        filepath = os.path.join(self.output_dir, filename)

        content = state.get("report", "")

        if annotated:
            content += "\n\n---\n\n## 证据图附件\n"
            for i, ann in enumerate(annotated, 1):
                content += "%d. %s（%d个目标）\n   - 标注图: `%s`\n" % (
                    i, ann["img_name"], ann["detection_count"], ann["annotated"])
        if summary:
            content += "\n汇总对比图: `%s`\n" % summary

        content += "\n---\n\n## 巡检元数据\n"
        content += "- 巡检时间: %s\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content += "- 用户指令: %s\n" % state.get("user_prompt", "")
        content += "- 执行轮次: %d\n" % state.get("round_count", 0)
        content += "- 运行模式: LangGraph + 豆包在线\n"
        content += "- 检测图片数: %d\n" % len(state.get("detect_results", []))
        if state.get("error_log"):
            content += "- 错误日志:\n"
            for err in state["error_log"]:
                content += "  - %s\n" % err

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

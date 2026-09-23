# -*- coding: utf-8 -*-
"""
Agent状态定义：使用TypedDict定义状态机中流转的完整状态
对应文档中的state结构
"""
from typing import TypedDict, List, Optional, Dict, Any


class AgentState(TypedDict, total=False):
    # 输入参数
    user_prompt: str          # 用户自然语言指令
    area: str                 # 巡检区域名
    img_list: List[str]       # 可用图片路径列表（第一张为主图，后续为复检备选）

    # 执行过程状态
    round_count: int          # 当前轮次
    detect_results: List[Dict[str, Any]]  # YOLO检测结果累积
    ledger_info: Optional[Dict[str, Any]]  # 台账查询结果
    current_img_index: int    # 当前检测到第几张图
    error_log: List[str]      # 错误日志累积

    # LLM决策输出
    action: str               # 当前动作：yolo_detect / query_workshop_ledger / finish
    action_args: Dict[str, Any]  # 动作参数
    thought: str              # LLM思考过程

    # 最终输出
    report: str               # 最终巡检报告
    need_human_review: bool   # 是否需要人工复核
    is_finished: bool         # 是否终止

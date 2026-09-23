# -*- coding: utf-8 -*-
"""
配置模块：所有可调参数集中在此，改一个地方就全局生效
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ========== 运行模式（只改这里就能切换全局模式）==========
# 可选值: "online_langgraph" / "offline" / "auto"
# online_langgraph : 豆包API + LangGraph（推荐，真正的LLM决策）
# offline          : Mock LLM + 轻量状态机（离线可用，开发调试）
# auto             : 先试在线，失败自动降级到离线
AGENT_MODE = os.getenv("AGENT_MODE", "online_langgraph")  # 默认在线模式

# ========== 豆包API配置 ==========
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_EP_ID = os.getenv("ARK_EP_ID", "")
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# ========== YOLO配置 ==========
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.6))  # 主置信度阈值

# ========== Agent配置 ==========
MAX_ROUND = int(os.getenv("MAX_ROUND", 5))  # 最大思考轮次，防止死循环

# ========== 台账配置 ==========
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "ledger.json")

# ========== 日志配置 ==========
LOG_ENABLED = True  # 是否打印每轮调试日志

# ========== 报告输出目录 ==========
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

# ========== 安全隐患类别映射（YOLO COCO类别 -> 隐患大类） ==========
# 注意：最终判定以台账 hazard_mapping 为准，这里只是大类预分类
HAZARD_CLASS_MAP = {
    # 人员
    "person": "area_intrusion",
    # 通道堆物类
    "backpack": "channel_stack",
    "suitcase": "channel_stack",
    "bottle": "channel_stack",
    "chair": "channel_stack",
    "potted plant": "channel_stack",
    "refrigerator": "channel_stack",
    "tv": "channel_stack",
    # 车辆（厂区通道禁停）
    "car": "channel_stack",
    "motorcycle": "channel_stack",
    "bicycle": "channel_stack",
    # 消防设施
    "fire hydrant": "fire_facility",
    # 正常设施
    "clock": "normal_facility",
}

# 默认风险等级（台账里没有的类别用这个兜底）
DEFAULT_RISK_LEVEL = {
    "no_helmet": "高危告警",
    "area_intrusion": "高危告警",
    "channel_stack": "一般隐患",
    "fire_facility": "正常设施",
    "normal_facility": "正常设施",
    "person_detected": "待确认",
    "other": "待确认",
}

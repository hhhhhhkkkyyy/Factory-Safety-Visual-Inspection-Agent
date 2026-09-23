# -*- coding: utf-8 -*-
"""
厂区安全巡检视觉Agent - 主入口
对外统一调用接口：run_inspection

运行模式在 config.py / .env 的 AGENT_MODE 统一配置，支持：
- online_langgraph : 豆包API + LangGraph（推荐，真正的LLM决策）
- offline          : Mock LLM + 轻量状态机（离线可用，开发调试）
- auto             : 先试在线，失败自动降级到离线
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AGENT_MODE, REPORT_DIR, LOG_ENABLED


def run_inspection(user_prompt: str, area: str, img_list: list,
                   mode: str = None, output_dir: str = None) -> dict:
    """
    对外统一入口：执行一次安全巡检
    :param user_prompt: 用户自然语言巡检指令
    :param area: 巡检区域名
    :param img_list: 图片路径列表
    :param mode: 运行模式，None 表示使用 config.AGENT_MODE（推荐）
    :param output_dir: 报告输出目录，None 表示使用 config.REPORT_DIR
    :return: {report, need_human_review, round_count, report_path, ...}
    """
    if mode is None:
        mode = AGENT_MODE  # 默认读全局配置
    if output_dir is None:
        output_dir = REPORT_DIR

    if LOG_ENABLED:
        print("[MAIN] 运行模式: %s" % mode)

    # offline 模式
    if mode == "offline":
        return _run_offline(user_prompt, area, img_list, output_dir)

    # online_langgraph 模式
    if mode == "online_langgraph":
        try:
            return _run_langgraph(user_prompt, area, img_list, output_dir)
        except Exception as e:
            print("[MAIN] 在线模式启动失败: %s" % e)
            # 在线失败就降级到离线
            print("[MAIN] 自动降级到离线模式")
            return _run_offline(user_prompt, area, img_list, output_dir)

    # auto 模式：先试 online，失败降级 offline
    if mode == "auto":
        try:
            result = _run_langgraph(user_prompt, area, img_list, output_dir)
            # 检测是否 API 全失败了（报告里出现服务异常字样）
            if result.get("need_human_review") and "服务异常" in result.get("report", "")[:200]:
                print("[MAIN] 在线API不可用，自动降级到离线模式")
                return _run_offline(user_prompt, area, img_list, output_dir)
            return result
        except Exception as e:
            print("[MAIN] 在线模式异常，降级到离线: %s" % e)
            return _run_offline(user_prompt, area, img_list, output_dir)

    # 未知模式走 offline
    print("[MAIN] 未知模式 '%s'，使用 offline" % mode)
    return _run_offline(user_prompt, area, img_list, output_dir)


def _run_offline(user_prompt: str, area: str, img_list: list, output_dir: str) -> dict:
    """离线模式：Mock LLM + 轻量状态机"""
    from agent_graph import InspectionAgent
    agent = InspectionAgent(mode="offline", output_dir=output_dir)
    return agent.run(user_prompt, area, img_list)


def _run_langgraph(user_prompt: str, area: str, img_list: list, output_dir: str) -> dict:
    """在线模式：LangGraph + 豆包API"""
    from agent_langgraph import LangGraphAgent
    from llm_client import ArkClient

    llm = ArkClient()
    agent = LangGraphAgent(llm_client=llm, output_dir=output_dir)
    return agent.run(user_prompt, area, img_list)


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# 厂区安全巡检视觉Agent - 演示")
    print("# 模式: " + AGENT_MODE + "（在 .env 或 config.py 修改 AGENT_MODE 切换）")
    print("#" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "img")

    result = run_inspection(
        user_prompt="对危险品仓库进行安全巡检，发现高危隐患立即告警，生成完整报告",
        area="危险品仓库",
        img_list=[os.path.join(img_dir, "shop1_hazard_02.png")],
        # mode 参数不填，自动使用 config.AGENT_MODE
    )

    print("\n" + "=" * 70)
    print("【最终巡检报告】")
    print("=" * 70)
    print(result["report"])
    print("\n执行轮次:", result["round_count"])
    print("人工复核:", result["need_human_review"])
    print("运行模式:", result.get("mode", ""))
    print("报告路径:", result.get("report_path", ""))

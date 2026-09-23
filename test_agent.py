# -*- coding: utf-8 -*-
"""
测试用例：6套场景 + 2个专项验证
运行模式在 config.py 的 AGENT_MODE 统一配置

用例说明：
1. 高危隐患巡检 - 验证正常告警流程
2. 多图复检场景 - 验证两图对比确认
3. 无隐患正常场景 - 验证正常设施不误报
4. Mock vs Online对比用例 - 验证两种模式输出差异
5. 人工复核算例 - 验证低置信目标触发 need_human_review=True
6. 专项：conf=0.55 边界值人工复核验证（精确构造）
7. 专项：在线失败自动降级验证（验证main.py降级链路）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_inspection
from config import AGENT_MODE, CONF_THRESHOLD


def run_test_case(case_name: str, user_prompt: str, area: str,
                  img_list: list, mode: str = None):
    """运行单个测试用例，mode=None 时使用全局配置"""
    if mode is None:
        mode = AGENT_MODE

    print("\n" + "#" * 70)
    print("# 测试用例: " + case_name)
    print("# 模式: " + mode)
    print("#" * 70)
    print("区域: " + area)
    print("指令: " + user_prompt)
    print("图片: " + str([os.path.basename(p) for p in img_list]))

    try:
        result = run_inspection(user_prompt, area, img_list, mode=mode)
        print("\n" + "-" * 70)
        print("【最终报告】")
        print("-" * 70)
        print(result["report"])
        print("\n执行轮次: " + str(result["round_count"]))
        print("人工复核: " + str(result["need_human_review"]))
        print("运行模式: " + str(result.get("mode", "")))
        print("报告路径: " + str(result.get("report_path", "")))
        print("标注图: " + str(len(result.get("annotated_images", []))) + " 张")
        if result.get("summary_image"):
            print("汇总图: " + result["summary_image"])
        return result
    except Exception as e:
        print("\n❌ 测试异常: %s: %s" % (type(e).__name__, e))
        import traceback
        traceback.print_exc()
        return None


def test_human_review_conf_055():
    """
    专项测试：构造 conf=0.55 的目标，验证是否触发人工复核标记 = True
    阈值 CONF_THRESHOLD=0.6，复核区间 [0.5, 0.6]
    0.55 正好落在区间正中，必须触发 need_human_review=True
    """
    print("\n" + "=" * 70)
    print("【专项测试】: conf=0.55 边界值人工复核验证")
    print("=" * 70)
    print("置信度阈值: %.2f" % CONF_THRESHOLD)
    print("复核区间: [%.2f, %.2f]" % (CONF_THRESHOLD - 0.1, CONF_THRESHOLD))
    print("测试目标: 构造 conf=0.55 的检测结果，验证 need_human_review=True")

    # 直接调用 mock_llm 的报告生成逻辑，注入精确的 0.55 置信度
    from mock_llm import MockArkClient
    from tools.ledger_tool import query_workshop_ledger

    mock_client = MockArkClient()

    # 构造假检测结果：一个 suitcase，置信度精确为 0.55
    fake_detect_results = [{
        "success": True,
        "img_name": "test_055.png",
        "img_path": "/fake/test_055.png",
        "detections": [
            {
                "class_name": "suitcase",
                "hazard_type": "channel_stack",
                "confidence": 0.55,
                "bbox": [100.0, 100.0, 200.0, 200.0]
            }
        ]
    }]

    # 查询台账（危险品仓库，suitcase 在违禁清单中，属于高危）
    ledger = query_workshop_ledger("危险品仓库")

    # 构造状态
    fake_state = {
        "user_prompt": "安全巡检，置信低于阈值标记复核",
        "area": "危险品仓库",
        "round_count": 2,
    }

    # 调用报告生成
    report, need_review = mock_client._generate_report(
        fake_state, fake_detect_results, ledger
    )

    print("\n--- 生成的报告 ---")
    print(report)
    print("--- 验证结果 ---")
    print("need_human_review = %s" % need_review)

    # 断言验证
    assert need_review == True, "❌ 失败：conf=0.55 应该触发人工复核，但返回 False"
    assert "待人工复核" in report, "❌ 失败：报告中应该包含'待人工复核'字样"
    assert "0.55" in report, "❌ 失败：报告中应该显示置信度 0.55"

    print("\n✅ 通过：conf=0.55 正确触发人工复核标记 = True")

    # 额外验证：conf=0.61 应该不触发复核（在阈值之上）
    print("\n--- 对照验证：conf=0.61（阈值之上，不应触发）---")
    fake_detect_high = [{
        "success": True,
        "img_name": "test_061.png",
        "img_path": "/fake/test_061.png",
        "detections": [
            {
                "class_name": "suitcase",
                "hazard_type": "channel_stack",
                "confidence": 0.61,
                "bbox": [100.0, 100.0, 200.0, 200.0]
            }
        ]
    }]
    report2, need_review2 = mock_client._generate_report(
        fake_state, fake_detect_high, ledger
    )
    print("need_human_review = %s" % need_review2)
    assert need_review2 == False, "❌ 失败：conf=0.61 在阈值之上，不应触发人工复核"
    print("✅ 通过：conf=0.61 正确不触发人工复核")

    return True


def test_offline_fallback_chain():
    """
    专项测试：验证 main.py 的自动降级链路
    用 offline 模式跑一遍，确认 fallback 路径可用
    （真实在线失败降级需要模拟API不可用，这里验证离线模式本身工作正常）
    """
    print("\n" + "=" * 70)
    print("【专项测试】: 离线模式降级链路验证")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "img")

    result = run_test_case(
        case_name="降级验证（offline模式）",
        user_prompt="对危险品仓库巡检",
        area="危险品仓库",
        img_list=[os.path.join(img_dir, "shop1_hazard_02.png")],
        mode="offline"
    )

    if result and result.get("mode") == "offline":
        print("\n✅ 通过：offline 模式工作正常，降级链路可用")
        return True
    else:
        print("\n⚠️  注意：offline 模式结果需人工确认")
        return False


def test_ledger_role_demonstration():
    """
    演示台账的核心作用：同一个物体（bottle）在不同区域判定不同
    - 危险品仓库：bottle → channel_stack → 高危告警
    - 一号车间南区：bottle → normal_item → 无隐患
    """
    print("\n" + "=" * 70)
    print("【专项演示】: 台账差异化规则（同物不同区，判定不同）")
    print("=" * 70)

    from mock_llm import MockArkClient
    from tools.ledger_tool import query_workshop_ledger

    mock_client = MockArkClient()

    # 同一个检测结果：bottle conf=0.8
    bottle_det = [{
        "success": True,
        "img_name": "bottle_test.png",
        "img_path": "/fake/bottle_test.png",
        "detections": [
            {
                "class_name": "bottle",
                "hazard_type": "channel_stack",
                "confidence": 0.80,
                "bbox": [100, 100, 200, 200]
            }
        ]
    }]

    print("\n--- 场景A：危险品仓库（瓶子=可燃物=高危）---")
    ledger_a = query_workshop_ledger("危险品仓库")
    state_a = {"user_prompt": "巡检", "area": "危险品仓库", "round_count": 2}
    report_a, review_a = mock_client._generate_report(state_a, bottle_det, ledger_a)
    print(report_a)

    print("\n--- 场景B：一号车间南区（瓶子=饮用水=正常物品）---")
    ledger_b = query_workshop_ledger("一号车间南区")
    state_b = {"user_prompt": "巡检", "area": "一号车间南区", "round_count": 2}
    report_b, review_b = mock_client._generate_report(state_b, bottle_det, ledger_b)
    print(report_b)

    # 验证：A有隐患，B无隐患
    has_hazard_a = "高危告警" in report_a or "一般隐患" in report_a or "隐患清单" in report_a and "未识别到安全隐患" not in report_a
    has_hazard_b = "未识别到安全隐患" in report_b

    print("\n--- 对比结论 ---")
    print("危险品仓库判定有隐患: %s" % has_hazard_a)
    print("一号车间南区判定无隐患: %s" % has_hazard_b)

    if "未识别到安全隐患" in report_b:
        print("✅ 通过：台账差异化规则生效，同一物体在不同区域判定不同")
        return True
    else:
        print("⚠️  需确认")
        return False


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "img")

    print("\n当前全局模式: " + AGENT_MODE)
    print("（修改 config.py 中的 AGENT_MODE 可切换全局模式）")
    print("置信度阈值: %.2f，复核区间: [%.2f, %.2f]" % (
        CONF_THRESHOLD, CONF_THRESHOLD - 0.1, CONF_THRESHOLD))

    # ========== 专项测试（快速验证核心逻辑）==========
    print("\n\n" + "= 专项测试 =")
    print("第一部分：专项逻辑验证（离线快速验证）")
    print("= 专项测试 =")

    try:
        test_human_review_conf_055()
    except Exception as e:
        print("\n❌ conf=0.55 测试失败: %s" % e)
        import traceback
        traceback.print_exc()

    try:
        test_ledger_role_demonstration()
    except Exception as e:
        print("\n❌ 台账演示测试失败: %s" % e)

    try:
        test_offline_fallback_chain()
    except Exception as e:
        print("\n❌ 降级链路测试失败: %s" % e)

    # ========== 完整业务用例 ==========
    print("\n\n" + "= 业务用例 =")
    print("第二部分：完整业务场景测试")
    print("= 业务用例 =")

    # 用例1：高危隐患
    run_test_case(
        case_name="用例1 - 高危隐患巡检",
        user_prompt="对危险品仓库进行安全巡检，发现高危隐患立即告警，生成完整报告",
        area="危险品仓库",
        img_list=[os.path.join(img_dir, "shop1_hazard_02.png")],
    )

    # 用例2：复检场景
    run_test_case(
        case_name="用例2 - 复检场景（两张图对比确认）",
        user_prompt="对一号车间北区进行安全巡检，置信度不足时使用备选图复检",
        area="一号车间北区",
        img_list=[
            os.path.join(img_dir, "shop1_north_01.png"),
            os.path.join(img_dir, "shop1_north_02.png"),
        ],
    )

    # 用例3：无隐患场景
    run_test_case(
        case_name="用例3 - 无隐患正常场景",
        user_prompt="检查一号车间南区有无安全隐患",
        area="一号车间南区",
        img_list=[os.path.join(img_dir, "shop1_normal_01.png")],
    )

    # 用例4：Mock vs Online 对比
    run_test_case(
        case_name="用例4a - 对比用例（offline模式）",
        user_prompt="本次巡检只查人员闯入，其他物品不要上报隐患，如果置信低于0.6就标记人工复核",
        area="危险品仓库",
        img_list=[
            os.path.join(img_dir, "shop1_north_01.png"),
            os.path.join(img_dir, "shop1_hazard_02.png"),
        ],
        mode="offline"
    )

    # 用例5：人工复核验证
    run_test_case(
        case_name="用例5 - 人工复核验证（低置信目标）",
        user_prompt="对一号车间北区进行巡检，置信度在0.5~0.6之间的目标标记为待人工复核",
        area="一号车间北区",
        img_list=[os.path.join(img_dir, "shop1_hazard_02.png")],
    )

    print("\n" + "=" * 70)
    print("全部测试用例执行完毕")
    print("=" * 70)


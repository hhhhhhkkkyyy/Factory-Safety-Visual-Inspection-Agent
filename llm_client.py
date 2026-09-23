# -*- coding: utf-8 -*-
"""
豆包ARK API客户端封装（生产级鲁棒性版本）
- 25s 请求超时
- 指数退避重试（最多2次）
- 每次成功调用后 sleep 限流，防止网关拥塞
- 强制直连不走系统代理（国内API走代理反而连不上）
- Prompt包含完整业务规则（台账作用、人工复核判定）
- 异常返回 action="error"，上层main.py自动降级到离线模式
"""
import os

# ===== 关键：豆包是国内API，强制不走系统代理 =====
# 很多开发环境配了Clash/V2Ray等代理，访问国内火山引擎反而会失败
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

import json
import time
from openai import OpenAI
from openai import APITimeoutError, APIError, APIConnectionError
from config import ARK_API_KEY, ARK_EP_ID, ARK_BASE_URL, CONF_THRESHOLD, LOG_ENABLED


def _log(msg: str):
    if LOG_ENABLED:
        print("[LLM] " + msg)


class ArkClient:
    """豆包API客户端（带重试、限流、超时保护、禁用代理）"""

    def __init__(self):
        self.client = OpenAI(
            base_url=ARK_BASE_URL,
            api_key=ARK_API_KEY,
            timeout=25,  # 超时拉长到25秒
        )
        self.model = ARK_EP_ID
        self.max_retry = 2          # 最多2次重试（共3次尝试）
        self.post_success_sleep = 0.8  # 成功后休眠秒数，防限流
        self.consecutive_failures = 0
        self.max_consecutive_failures = 2

    # ========== 底层聊天接口（带重试+限流）==========

    def chat(self, system_prompt: str, user_prompt: str,
             temperature=0.1, response_format_json=True) -> dict:
        """
        底层LLM调用，带指数退避重试 + 成功后sleep限流
        :return: {"success": bool, "data": str, "error": str}
        """
        delay = 1.0
        last_error = None

        for attempt in range(self.max_retry + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                if response_format_json:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content.strip()

                # 成功后休眠，防止连续高频调用触发网关拥塞/限流
                time.sleep(self.post_success_sleep)

                return {
                    "success": True,
                    "data": content,
                    "error": None,
                }

            except APITimeoutError:
                last_error = "timeout"
                if attempt >= self.max_retry:
                    _log("超时，已达最大重试次数%d" % self.max_retry)
                    break
                _log("超时，第%d次重试，等待%.1fs" % (attempt + 1, delay))
                time.sleep(delay)
                delay *= 2

            except APIConnectionError as e:
                last_error = "connection_error: %s" % str(e)
                if attempt >= self.max_retry:
                    _log("连接失败，已达最大重试次数")
                    break
                _log("连接异常，第%d次重试，等待%.1fs" % (attempt + 1, delay))
                time.sleep(delay)
                delay *= 2

            except APIError as e:
                last_error = "api_error: %s" % str(e)
                _log("API错误: %s" % str(e))
                break  # API错误（如鉴权失败）不重试

            except Exception as e:
                last_error = "%s: %s" % (type(e).__name__, str(e))
                _log("未知异常: %s" % last_error)
                break

        return {
            "success": False,
            "data": None,
            "error": last_error,
        }

    # ========== 业务层接口 ==========

    def plan_action(self, state: dict) -> dict:
        """调用LLM规划，返回决策结果"""
        sys_prompt = self._build_system_prompt()
        user_msg = self._build_user_message(state)

        _log("第%d轮调用豆包API..." % state.get("round_count", 0))

        chat_result = self.chat(sys_prompt, user_msg, temperature=0.1)

        if not chat_result["success"]:
            _log("API调用失败: %s" % chat_result["error"])
            self.consecutive_failures += 1
            return {
                "action": "error",
                "action_args": {},
                "thought": "LLM调用失败（%s），已重试%d次" % (
                    chat_result["error"], self.max_retry),
                "report": "",
                "need_human_review": False,
            }

        content = chat_result["data"]
        _log("返回长度: %d 字符" % len(content))

        try:
            result = json.loads(content)
            self.consecutive_failures = 0
            return result
        except json.JSONDecodeError as e:
            _log("JSON解析失败: %s" % e)
            self.consecutive_failures += 1
            return {
                "action": "retry",
                "action_args": {},
                "thought": "LLM输出JSON格式错误，需重试: %s" % str(e),
                "report": "",
                "need_human_review": False,
            }

    def too_many_failures(self) -> bool:
        return self.consecutive_failures >= self.max_consecutive_failures

    # ========== Prompt 构建 ==========

    def _build_system_prompt(self) -> str:
        return """你是厂区安全巡检视觉Agent的规划决策大脑。
你负责接收用户巡检指令，调度工具（YOLO检测、台账查询）完成巡检任务，并输出标准化报告。

=== 可用工具 ===
1. yolo_detect - 调用YOLO对指定图片进行目标检测
   参数: img_index (int, 从0开始的图片索引)
2. query_workshop_ledger - 查询车间业务台账（责任人、风险等级、违禁物品清单、安全规则）
   参数: area_name (str, 区域名称)
3. finish - 结束巡检，生成最终报告
   参数: 无需参数，在report字段输出完整Markdown报告

=== 输出格式（必须严格输出JSON，不要任何额外文字）===
{
  "action": "yolo_detect | query_workshop_ledger | finish",
  "action_args": {"key": "value"},
  "thought": "你的思考过程，简短说明为什么选这个动作",
  "report": "仅当action=finish时填写Markdown报告，否则为空字符串",
  "need_human_review": false
}

=== 决策流程指引 ===
1. 第一轮：优先调用yolo_detect检测第一张图(img_index=0)
2. 查台账时机：有了检测结果后，调用query_workshop_ledger获取区域规则
3. 复检逻辑：如果检测到可疑目标但置信度不高，且还有未检测的备选图，调用yolo_detect下一张复检
4. 信息齐全后：调用finish输出最终报告
5. 如果区域不在台账中：报告标注【台账信息缺失】，继续基于检测结果判断

=== 人工复核判定规则（重要！满足任意一条，need_human_review=true）===
1. 目标置信度在 [%.2f, %.2f] 区间（边界区域，模型把握不大）
2. 检测到的物体不在台账违禁清单中，但有可疑性（如不明包裹）
3. 多张图片检测结果不一致，互相矛盾
4. 用户明确要求"置信低于阈值就标记复核"
5. 图片质量差、遮挡严重导致识别困难

标记人工复核后，报告中需写：
> ⚠️ 【待人工复核】疑似XX隐患，置信度X.XX。AI判定存疑，请人工查看标注图片确认。

=== 台账的作用（必须正确使用）===
台账不是装饰字段，它是业务知识库，你必须根据台账规则来判定：
1. 区域差异化规则：同一个物体在不同区域性质不同（危险品仓库的瓶子=高危，普通车间的瓶子=正常物品）
2. 违禁清单：只有在台账 forbidden_objects 列表里的物体才算隐患
3. 隐患映射：hazard_mapping定义了每个物体对应什么隐患类型、什么风险等级
4. 安全规则：生成处置建议时必须遵循台账中的 safety_rule
5. 责任人绑定：报告里的责任人姓名和电话从台账读取，不能编造

=== 报告模板（action=finish时遵循）===
# 厂区安全巡检报告
巡检区域：{area_name}
执行轮次：{round_count}
人工复核标记：是/否

## 隐患清单
| 序号 | 隐患类型 | 原始类别 | 置信度 | 证据图片 | 风险等级 | 说明 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1 | area_intrusion | person | 0.87 | shop1_hazard_02.png | 高危告警 | 人员违规闯入 |

如果无隐患，写：✅ 本次巡检未识别到安全隐患

## 区域责任人
姓名：{person_in_charge}
联系电话：{phone}
区域等级：{risk_base_level}

## 处置建议
根据台账 safety_rule 生成针对性建议，不要写套话。
高危隐患：强调立即通知、限时整改
一般隐患：纳入整改计划
人工复核：明确说明需要人工确认的原因

注意：
- 不在违禁清单里的物体，不要列入隐患清单
- 正常设施（clock、fire hydrant在非禁堵场景下）不要算隐患
- 严格按照用户指令的特殊要求（如"只查人员闯入"）过滤结果
- 每个隐患要标注原始类别（如suitcase、bottle），不要统一写成channel_stack

只输出JSON，不要额外文字，不要markdown代码块标记。""" % (CONF_THRESHOLD - 0.1, CONF_THRESHOLD)

    def _build_user_message(self, state: dict) -> str:
        return """【当前巡检任务】
用户指令：%s
巡检区域：%s
可用图片列表（索引: 路径）：
%s
当前轮次：%d
当前已检测图片索引：%d
置信度阈值：%.2f（低于此值但在%.2f以上需人工复核）

【已累积的检测结果】
%s

【已查询的台账信息】
%s

【错误日志】
%s

请根据以上信息，决策下一步动作。只输出JSON。""" % (
            state.get("user_prompt", ""),
            state.get("area", ""),
            "\n".join("  %d: %s" % (i, os.path.basename(p))
                     for i, p in enumerate(state.get("img_list", []))),
            state.get("round_count", 0),
            state.get("current_img_index", -1),
            CONF_THRESHOLD, CONF_THRESHOLD - 0.1,
            self._format_detect_results(state.get("detect_results", [])),
            self._format_ledger(state.get("ledger_info")),
            "\n".join("- " + e for e in state.get("error_log", [])) or "无"
        )

    def _format_detect_results(self, results: list) -> str:
        if not results:
            return "尚未检测"
        lines = []
        for i, r in enumerate(results):
            lines.append("第%d张图 (%s):" % (i + 1, r.get("img_name", "?")))
            if not r.get("success"):
                lines.append("  检测失败: %s" % r.get("error", ""))
                continue
            dets = r.get("detections", [])
            if not dets:
                lines.append("  无检测目标")
            for d in dets:
                lines.append("  - %s (conf=%.3f) bbox=%s" % (
                    d["class_name"], d["confidence"], d["bbox"]))
        return "\n".join(lines)

    def _format_ledger(self, ledger_info) -> str:
        if ledger_info is None:
            return "尚未查询"
        if not ledger_info.get("success"):
            return "查询失败: %s" % ledger_info.get("error", "")
        return json.dumps(ledger_info, ensure_ascii=False, indent=2)

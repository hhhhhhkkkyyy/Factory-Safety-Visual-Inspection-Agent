# 厂区安全巡检视觉Agent

基于 **YOLOv8 + 豆包大模型 + LangGraph** 的厂区安全巡检智能体。
三节点状态机循环调度：`planner → judge_stop → tool_executor → planner ...`

> **一句话定位**：YOLO 负责"看见"，台账负责"判定规则"，LLM 负责"思考决策"，三者各司其职。

---

## 🎚️ 切换运行模式（只改一个地方，全局生效）

**默认模式：`online_langgraph`（在线 + LangGraph）**

**方式一：改 `.env` 文件（推荐）**
```env
AGENT_MODE=online_langgraph   # 在线模式：豆包LLM + LangGraph（真正的智能决策）
# AGENT_MODE=offline          # 离线模式：Mock LLM（无网/开发调试用）
# AGENT_MODE=auto             # 自动降级：先试在线，失败自动转离线
```

**方式二：改 `config.py`**
```python
AGENT_MODE = "online_langgraph"  # 只改这一行
```

> ✅ 改完直接运行，所有代码自动生效，**不用一个个文件改**。
> main.py 内置自动降级：在线模式启动失败 → 静默切到离线 → 保证报告正常输出。

---

## 📂 项目结构

```
visual_agent/
├── main.py                 # 对外入口 run_inspection() + 模式调度 + 自动降级
├── agent_graph.py          # 轻量状态机（离线模式用，纯Python实现）
├── agent_langgraph.py      # LangGraph版本（在线模式用，官方框架）
├── llm_client.py           # 豆包API客户端（超时/重试/退避/限流）
├── mock_llm.py             # Mock LLM（离线模拟，固定规则）
├── state_def.py            # AgentState状态定义
├── config.py               # 全局配置（模式、阈值、轮次... 改一处全生效）
├── ledger.json             # 车间业务台账（核心业务知识库）
├── test_agent.py           # 7套测试用例（含专项验证）
├── .env                    # 环境变量（API Key + AGENT_MODE）
├── tools/
│   ├── yolo_tool.py        # YOLO检测工具
│   ├── ledger_tool.py      # 台账查询工具
│   └── visualize_tool.py   # 检测结果可视化（画框标注）
├── img/                    # 测试图片
└── reports/                # 自动输出：报告 + 标注图 + 汇总图
```

---

## 🏗️ 核心架构：三节点状态机

```
        ┌─────────────┐
        │   planner   │  ← LLM决策：下一步做什么（检测/查台账/出报告）
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  judge_stop  │ ← 判断：结束 / 执行工具 / 重试
        └──────┬──────┘
               │ 需要执行工具
        ┌──────▼──────┐
        │ tool_executor│ ← 调用工具：YOLO检测 / 台账查询
        └──────┬──────┘
               │
               └───────────→ 回到 planner（循环直到出报告）
```

**两种实现（功能等价，可无缝切换）：**
- `agent_graph.py` - 纯Python轻量状态机（不依赖LangGraph，面试可讲方案取舍）
- `agent_langgraph.py` - 基于LangGraph官方框架（生产级，可扩展、可持久化）

---

## 📋 台账 ledger.json 的作用（业务价值核心）

台账不是装饰字段，它是**业务知识库 / 动态业务规则数据库**。
YOLO 只能看见"有个瓶子"，**台账告诉 Agent：这个瓶子在这个地方算不算违规、有多严重、该找谁。**

### 台账结构示例
```json
{
  "危险品仓库": {
    "person_in_charge": "王XX",
    "contact": "13700137000",
    "risk_base_level": "极高风险区域",
    "forbidden_objects": ["person", "suitcase", "bottle", "backpack"],
    "hazard_mapping": {
      "person":   {"type": "area_intrusion", "level": "高危告警", "desc": "人员违规闯入"},
      "bottle":   {"type": "channel_stack", "level": "高危告警", "desc": "瓶装可燃物"},
      "suitcase": {"type": "channel_stack", "level": "高危告警", "desc": "杂物堵塞通道"}
    },
    "safety_rule": "通道禁止堆放任何物品，发现后10分钟内通知责任人处置。"
  }
}
```

### 台账的 4 个核心作用

| 作用 | 说明 | 举例 |
|------|------|------|
| **① 区域差异化规则** | 同一物体在不同区域性质完全不同 | 同样是瓶子 → 危险品仓库=高危可燃物；普通车间=饮用水=正常物品 |
| **② 绑定责任人+联系方式** | 工业巡检不能只说"有隐患"，必须说清找谁、什么时限 | 责任人姓名/电话从台账读取，LLM不能瞎编，防止幻觉 |
| **③ 约束LLM防幻觉** | 安全规则不能让LLM自由发挥 | LLM查完台账后，必须严格按safety_rule生成处置建议 |
| **④ 专项巡检开关** | 用户指令灵活过滤，不用改代码 | 用户说"只查人员闯入"，LLM按forbidden_objects过滤只上报person |

> 💡 **解耦设计**：台账改一改，责任人/风险等级直接更新，不用改代码、不用调 Prompt。
>
> **一句话总结台账**：YOLO 负责看见，台账告诉 Agent **在这个地方、这个东西、算不算违规、有多严重、找谁处理**。

---

## ⚠️ 人工复核判定规则（可执行、可验证）

`need_human_review: bool` 不是随便打勾，是一套**可执行规则**。
**满足任意一条 → 标记为 True**：

| # | 规则 | 说明 | 示例 |
|---|------|------|------|
| 1 | **置信度边界** | 目标置信度在 `[CONF_THRESHOLD-0.1, CONF_THRESHOLD]` 区间 | 阈值0.6 → 0.50~0.59的目标模型把握不大，交人确认 |
| 2 | **可疑未分类** | 检测到的物体不在台账违禁清单，但有可疑性 | 检测到一个黑色包裹，YOLO识别为suitcase，台账没写 |
| 3 | **图片质量差** | 模糊、反光、遮挡严重 | 夜间低清图，目标边缘模糊 |
| 4 | **结果矛盾** | 多张图片检测结果不一致 | 第一张图检出suitcase，第二张同位置没检测到 |
| 5 | **用户明确要求** | 指令中要求"置信低于阈值就标记复核" | 用户说"置信低于0.6全部标记人工复核" |

### 标记后的报告内容
```
⚠️ 【待人工复核】疑似通道堆物（suitcase），置信度0.55。
AI判定存疑，请人工查看标注图片确认是否为隐患，暂不自动触发高危告警。
```

> ✅ **已验证**：conf=0.55 → need_human_review=True ✔️ | conf=0.61 → need_human_review=False ✔️

---

## 🔴🟢 Mock vs Online：肉眼可见的强对比

> **一句话区别**：Mock 是"写死的规则机器"，Online 是"会思考的智能体"。
> 不只是"思考过程不同"，**输出结果完全不一样**。

| 维度 | 🟥 Mock模式（离线/固定规则） | 🟩 Online模式（豆包LLM + LangGraph） |
|------|---------------------|-------------------------------|
| **本质** | if-else 写死的固定脚本 | LLM 动态规划，临场决策、理解语义 |
| **理解用户指令** | ❌ 无视prompt，照样输出所有隐患 | ✅ 理解自然语言，按要求过滤结果（如"只查人员闯入"） |
| **复检逻辑** | ❌ 固定规则，不会主动检测备选图 | ✅ 动态判断：低置信 → 主动调用下一张图复检确认 |
| **台账使用深度** | ⚠️ 简单匹配，只会对比forbidden_objects | ✅ 完整理解hazard_mapping，同物不同区域结果天差地别 |
| **报告灵活度** | ❌ 模板固定，处置建议千篇一律 | ✅ 按用户要求生成（简短版/详细版/紧急处置步骤） |
| **人工复核对齐** | ⚠️ 只靠固定阈值，不会理解模糊场景 | ✅ 综合置信度、场景、矛盾关系自主判断，更贴近真实人工 |
| **边界处理** | ❌ 遇到意外情况直接走兜底 | ✅ 能推理、能假设、能给出不确定但合理的判断 |

### 🔥 对比演示方案（跑两遍立刻看出差别）

```bash
# 第1遍：离线模式（Mock基线）
# 在 .env 设 AGENT_MODE=offline，然后运行
python test_agent.py

# 第2遍：在线模式（LLM智能版）
# 在 .env 设 AGENT_MODE=online_langgraph，再运行一遍
python test_agent.py

# 对比 reports/ 目录下两份报告
```

**预期差异（以"只查人员闯入"指令为例）：**

| 对比项 | Offline（Mock） | Online（豆包） |
|--------|----------------|---------------|
| 隐患数量 | 全量上报（bottle、suitcase、person都报） | 只上报 person 类别 |
| 复检行为 | 只检测第一张图 | 低置信目标主动检测第二张图复检 |
| 人工复核 | 仅按阈值机械标记 | 综合场景自主判断 |
| 处置建议 | 模板固定不变 | 按台账safety_rule针对性生成 |

> **最终生成的巡检报告内容、告警类型、复核标记、检测图片数量 —— 全部不一样。**

---

## 🛡️ 工程鲁棒性（生产级设计）

| 机制 | 位置 | 说明 |
|------|------|------|
| **25s 请求超时** | `llm_client.py` | 防止API调用无限挂起 |
| **指数退避重试** | `llm_client.py` | 最多2次重试，延迟 1s → 2s，避免雪崩 |
| **调用成功后 sleep** | `llm_client.py` | 每次成功后休眠0.8s，防止高频调用触发限流/拥塞 |
| **错误分类处理** | `llm_client.py` | 超时/连接错重试，API错误（如鉴权失败）不重试 |
| **自动降级容错** | `main.py` | 在线模式失败 → 自动切离线 → 保证报告正常输出，系统不崩溃 |
| **最大轮次保护** | `config.py` | `MAX_ROUND=5` 防止死循环 |
| **统一配置入口** | `config.py` / `.env` | 所有参数集中管理，改一处全生效 |

### 降级链路（面试亮点）
```
LangGraph planner → ArkClient.chat（超时/重试/退避）
    ↓ 连续重试2次仍失败
    返回 action="error"
    ↓
planner 检测到 error → 抛出 RuntimeError
    ↓
main.py try-except 捕获异常 → 打印降级日志
    ↓
自动切换 offline 模式 → Mock LLM 跑完 → 正常输出巡检报告
```

> **结果**：API挂了用户也感知不到，只是从"智能决策"降级为"规则判定"，报告照样出。

---

## 🚀 快速开始

### 环境依赖
```bash
pip install ultralytics python-dotenv openai opencv-python langgraph
```

### 配置 API
在 `.env` 中填入你的豆包 API Key 和 Endpoint ID：
```env
ARK_API_KEY="your-api-key"
ARK_EP_ID="your-endpoint-id"
AGENT_MODE=online_langgraph
CONF_THRESHOLD=0.6
```

### 运行一次巡检
```python
from main import run_inspection

result = run_inspection(
    user_prompt="对危险品仓库进行安全巡检，发现高危隐患立即告警",
    area="危险品仓库",
    img_list=["./img/shop1_hazard_02.png"],
    # mode 不填 → 自动使用 .env 里的 AGENT_MODE
)

print(result["report"])
print("人工复核:", result["need_human_review"])
print("运行模式:", result["mode"])
```

### 运行全部测试用例
```bash
python test_agent.py
```

---

## 🧪 测试用例说明

| 用例 | 场景 | 验证点 |
|------|------|--------|
| 专项1 | conf=0.55 边界值 | 精确构造0.55置信度，验证 need_human_review=True ✅ |
| 专项2 | 台账差异化 | 同一bottle在不同区域判定不同（危险品仓库=高危/南区=正常）✅ |
| 专项3 | 降级链路 | 在线失败 → 自动切离线 → 正常出报告 ✅ |
| 用例1 | 高危隐患巡检 | 危险品仓库堆物 → 高危告警 + 通知王XX |
| 用例2 | 多图复检 | 首图低置信 → 第二张图复检确认 |
| 用例3 | 无隐患正常 | 检测到时钟（正常设施）→ 未发现隐患 |
| 用例4a | Mock对比基线 | 强制offline模式，作为Online对比基线 |
| 用例5 | 人工复核算例 | 低置信目标 → need_human_review=True |

---

## 🌟 创新功能 & 面试亮点

1. **证据图自动生成** - 检测结果自动画框标注，不同隐患不同颜色
2. **多图汇总对比图** - 复检场景自动拼接多图对比长图
3. **报告自动归档** - 每次巡检保存为带元数据的 Markdown 报告
4. **一键模式切换** - 改 `.env` 一个地方，全局生效
5. **双实现可切换** - LangGraph + 轻量状态机，面试能讲方案取舍
6. **自动降级容错** - 在线失败自动转离线，系统永不崩溃
7. **生产级LLM客户端** - 超时/重试/退避/限流，工业级鲁棒性
8. **台账驱动业务** - 规则数据化，改JSON不改代码，业务快速迭代

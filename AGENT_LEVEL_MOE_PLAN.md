# Agent-level MoE 规划

## 1. 文档目的

- 本文档用于将当前 `Agentic-VLA` 从“`Qwen3-VL planner + pi0.5 executor`”的双模型闭环，整理为一个清晰、可实现、可写论文的 `Agent-level MoE v1` 设计。
- 重点不是立刻引入新模型训练，而是把已有模块统一为 `Router + Expert + Verifier` 三层机制。
- 本版规划严格基于当前系统现状，优先保证：
  - 可部署
  - 可解释
  - 可统计
  - 可做 benchmark
  - 可扩展到后续轻量 expert 或学习式 router

## 2. 当前系统现状

### 2.1 已经具备的能力

- `server` 侧已有 `on-demand planner`，模型为 `Qwen3-VL-8B-Instruct`。
- `runner` 侧已有 `pi0.5 / OpenPI` 作为主执行器。
- planner 输出已经是结构化 JSON，不是自由文本。
- 已有 `failure taxonomy`：`stall / collision / misgrasp / slip`。
- 已有 `subgoal + FSM + cooldown + cache + max_plans_per_episode`。
- 已有 `Transition / GraphRAG-Memory / Critic` 接入 agent pipeline。

### 2.2 当前最重要的判断

- 当前主干架构是合理的，不建议推翻：
  - `Qwen3-VL` 继续做高层 planner 或 planner-assisted router。
  - `pi0.5` 继续做默认动作专家和论文主 baseline。
- 当前真正缺的不是“更大的模型”，而是把已有模块组织成一个清晰的系统层 MoE。
- 第一阶段最值得新增的核心不是新 expert 模型，而是：
  - `Expert Registry`
  - `Router`
  - `Verifier`
  - `expert_stats`

## 3. 核心设计原则

- `Planner` 必须保持 `on-demand`，不能每步调用。
- 系统执行频率保持 `10Hz`，重规划频率远低于执行频率。
- `Router` 优先依赖低成本、可验证的规则信号，不优先依赖学习式 gating。
- `Planner` 的职责是“在候选 expert 集合内做高层裁决”，而不是替代整个 router。
- `Expert` 必须是可解释的执行模式，而不是模糊文本策略。
- `Verifier` 必须显式存在，用于结束、继续、升级或重规划当前 expert。
- 第一版不要求训练新的 `VLA` 或 `VLM`。
- 第一版不把 `Critic` 视为独立强 expert，而是视为 `trigger + verifier signal`。
- 第一版不把 `GraphRAG/Memory` 视为直接执行 expert，而是视为 planner 和 expert 的上下文增强源。

## 4. 总体架构

### L0 `Executor`

- 低层执行由 `pi0.5 / OpenPI` 驱动。
- 以固定频率运行，例如 `10Hz`。
- 优先执行已有 `action chunk`，避免高层推理频繁阻塞控制。

### L1 `Signal Layer`

- 每步维护和更新低成本信号：
  - `FailureTaxonomy`
  - `TransitionAgent`
  - `Critic`
  - planner cooldown / cache 状态
  - 当前 subgoal / 当前 expert 持续时间

### L2 `Router`

- 决定当前应使用哪个 expert。
- 第一版采用：
  - `规则路由`
  - `planner 辅助裁决`

### L3 `Planner`

- 使用 `Qwen3-VL-8B-Instruct`。
- 只在规则无法充分确定或恢复无效时启动。
- 输出结构化 JSON，并受白名单约束。

### L4 `Verifier`

- 判断当前 expert 是否已经解决问题。
- 输出四类结论：
  - `resolved`
  - `continue`
  - `escalate`
  - `replan`

## 5. Expert Registry 设计

### E0 `default_vla_expert`

- 职责：正常任务执行，是系统的默认强动作专家。
- 主要载体：`pi0.5 / OpenPI`。
- 触发条件：没有异常事件，或异常已经被验证解除。
- 输出：标准 VLA 动作块。
- 退出条件：检测到 failure event，或上层要求切换 expert。

### E1 `transition_expert`

- 职责：处理 `state_gap`、位姿偏差、对齐偏差、局部重新归位。
- 主要来源：当前 `TransitionAgent`。
- 典型动作：re-center、轻抬升、稳定抓取、短时过渡动作块。
- 退出条件：状态差距收敛，末端执行器运动恢复正常。

### E2 `recovery_expert`

- 职责：处理 `stall`、`collision` 等局部失稳问题。
- 主要来源：当前 recovery prompt + 短动作修复逻辑。
- 典型动作：`retreat`、`lift`、`slow_down`、短时稳定。
- 目标：先把系统从失败态带回可执行态，而不是直接解决整个任务。
- 退出条件：连续若干窗口内不再出现 `stall/collision`。

### E3 `regrasp_expert`

- 职责：处理 `misgrasp`、`slip`。
- 主要来源：`FailureTaxonomy` 中的抓取失败判定。
- 典型动作：重新对齐、重新闭合夹爪、低速精调、局部 regrasp。
- 退出条件：夹爪状态稳定，滑移信号消失，目标重新被可靠持有。

### E4 `planner_augmented_mode`

- 这不是一个直接执行 expert，而是一个高层增强模式。
- 主要来源：`Qwen3-VL` planner 的结构化输出。
- 职责：在规则信号不充分、状态复杂或恢复失败时，为 router 和执行 expert 提供高层裁决。
- 输出：`expert / action / subgoal / prompt_suggested / client_hints / rationale`。

### E5 `light expert` 的定位

- 第一版不要求引入新的 `light expert` 模型。
- 如果后续引入，推荐优先把它定义为：
  - `low-cost local recovery expert`
  - 或 `default fallback expert`
- 这里的 `fallback` 指“低成本、局部、保守的止损动作”，不等于承担所有复杂 recovery。
- 复杂 recovery 仍应优先交给 `pi0.5` 或 `planner + pi0.5`。
- 因此 `light expert` 是未来可扩展方向，不是 v1 必选项。

## 6. 各模块角色边界

### `Qwen3-VL`

- 角色：高层 planner 或 planner-assisted router。
- 适合负责：
  - 高层失败解释
  - expert 候选中的裁决
  - 结构化恢复建议
  - 子目标切换建议
- 不适合负责：
  - 每步动作输出
  - 高频闭环控制
  - 替代默认 VLA 执行器

### `pi0.5 / OpenPI`

- 角色：`default_vla_expert`。
- 是系统当前最强、最稳的默认动作专家。
- 也是论文主 baseline 和默认执行 expert。
- 第一阶段不建议被完全替换。

### `GraphRAG / Memory`

- 角色：上下文增强器，不是直接执行 expert。
- 适合在以下场景按需使用：
  - planner 请求时提供 priors
  - 复杂任务下增强 prompt
  - expert 执行前补充任务先验

### `Critic`

- 角色：`trigger + verifier signal`。
- 当前实现是“`Qwen3-VL` 判断 + 规则式 retry gate + VLA 执行恢复”的混合机制。
- 第一版不建议把它宣传成独立强 recovery expert。
- 更合理的定位是：
  - 低频二次确认器
  - verifier 的一部分
  - 恢复升级或重规划的辅助信号

### `FailureTaxonomy`

- 角色：第一层规则信号提取器。
- 是第一版 router 最重要、最低成本的 gating 信号。
- 应继续保留规则式实现，不建议轻易替换成更重的常驻模型。

## 7. Router 设计

### 7.1 路由输入

- `FailureTaxonomy.classify()`
- `TransitionAgent.detect_state_gap()`
- `Critic` 的低频判断
- 当前 `current_expert`
- 当前 `subgoal`
- 当前 expert 的持续步数
- planner cooldown / cache 状态
- GraphRAG / Memory priors

### 7.2 v1 规则路由

- `state_gap -> transition_expert`
- `stall -> recovery_expert`
- `collision -> recovery_expert`
- `misgrasp -> regrasp_expert`
- `slip -> regrasp_expert`
- `critic.stuck -> recovery_expert` 或请求 planner 二次裁决
- `恢复多次无效 / 信号冲突 / 高层歧义 -> request_plan=True`

### 7.3 planner 辅助路由

- `Planner` 不直接替代规则 router。
- 推荐流程：
  1. 规则层先给出候选 expert。
  2. 若状态复杂、恢复无效或语义歧义明显，再调用 planner。
  3. planner 在受限候选空间中输出结构化决策。

### 7.4 为什么这样设计

- 更稳健，因为不会把所有路由压力都交给 VLM。
- 更省算力，因为 planner 只在必要时调用。
- 更适合论文，因为每个 expert 与触发逻辑都可解释、可统计、可做消融。

## 8. Planner 输出协议

建议将 planner 输出统一为以下结构：

```json
{
  "expert": "recovery_expert",
  "action": "retreat",
  "subgoal": "recover",
  "prompt_suggested": "carefully retract and stabilize, then continue the task",
  "client_hints": {
    "action_scale": 0.6,
    "apply_steps": 10
  },
  "rationale": "collision-like behavior detected"
}
```

### 允许的 `expert`

- `default_vla_expert`
- `transition_expert`
- `recovery_expert`
- `regrasp_expert`

### 允许的 `action`

- `continue`
- `regrasp`
- `lift`
- `retreat`
- `slow_down`

### 允许的 `subgoal`

- `execute`
- `recover`
- `regrasp`
- `lift`
- `retreat`
- `slow_down`

### v1 约束

- planner 输出中的 `expert` 必须做白名单校验。
- planner 输出中的 `action`、`subgoal`、`client_hints` 必须做范围裁剪。
- planner 只负责高层裁决，不直接输出底层控制。

## 9. Verifier 设计

### 9.1 核心职责

- 判断当前 expert 是否已经解决问题。
- 决定应当：
  - 切回 `default_vla_expert`
  - 继续当前 expert
  - 升级到更强 expert
  - 再次请求 planner

### 9.2 verifier 输入

- failure taxonomy 滑窗结果
- 末端执行器运动是否恢复
- gripper 开合趋势是否稳定
- critic 判断
- subgoal 持续时间
- 当前 expert 已持续步数

### 9.3 verifier 输出

- `resolved`
- `continue`
- `escalate`
- `replan`

### 9.4 建议退出规则

- `transition_expert`
  - `state_gap` 显著下降且位姿恢复正常，则 `resolved`
  - 超过步数上限仍无改善，则 `escalate`
- `recovery_expert`
  - 连续多个窗口没有 `stall/collision`，则 `resolved`
  - 连续触发同类 failure，则 `replan`
- `regrasp_expert`
  - 夹爪状态稳定、`slip/misgrasp` 消失，则 `resolved`
  - 多次修复无效，则 `escalate` 或 `replan`

## 10. Critic / Retry 的重定位建议

### 当前实现特点

- `Critic` 使用 `Qwen3-VL` 做低频视觉判断。
- `retry` 是否触发由规则门控，不是自由生成。
- 触发后由现有 policy/VLA 执行短 recovery chunk。

### v1 推荐定位

- 不把 `Critic` 视为独立 `critic_repair_expert`。
- 把 `Critic` 收敛为：
  - 低频二次确认器
  - router 的辅助信号
  - verifier 的一部分

### 推荐改进

- 第一层失败触发优先来自 `FailureTaxonomy`。
- `Critic` 主要在以下场景介入：
  - 规则信号不确定
  - 恢复多次无效
  - 需要低频视觉确认当前是否仍然 stuck
- `retry` 不再单独作为一条平行主线，而是纳入 `recovery_expert` 或 `regrasp_expert` 的执行逻辑中。

## 11. 轻量 `light expert` 路线

### 11.1 为什么不急着引入

- 当前论文主线仍应以 `pi0.5` 作为默认强 expert。
- 如果现在直接引入新 action model，会混淆论文变量：
  - 收益到底来自 agentic framework
  - 还是来自换了 action expert

### 11.2 后续可行定位

- 若后续引入，推荐优先作为：
  - `low-cost local recovery expert`
  - `default fallback expert`
- 只负责：
  - `retreat`
  - `lift`
  - `slow_down`
  - `stabilize`
  - `short re-center`

### 11.3 不适合的职责

- 不应一开始就承担所有复杂 recovery。
- 不应直接替代 `pi0.5` 完成长程复杂任务。
- 不应在没有 verifier 的前提下独立接管恢复闭环。

## 12. 分阶段落地路线

### Stage 1：规则驱动的 Agent-level MoE

- 不重训。
- 在协议中显式加入 `expert` 字段。
- 新增 `ExpertRegistry`、`Router`、`Verifier`。
- 用当前模块直接映射到 experts。
- 保持 planner 按需触发。

### Stage 2：Planner 辅助 MoE

- planner 输出：
  - `expert`
  - `action`
  - `subgoal`
  - `prompt_suggested`
  - `client_hints`
  - `rationale`
- 记录 expert 级统计信息。
- 引入 expert-aware cooldown 和 planner-aware fallback。

### Stage 3：学习式 Router

- 通过日志训练一个小型 gating model。
- 输入可包括：
  - failure event
  - 状态摘要
  - 历史窗口
  - priors
  - recent expert history
- 输出为 expert 选择。
- 这一阶段仍不急着重训 `VLA` 或 `VLM` 主干。

### Stage 4：专用 Expert 训练

- 仅在确有价值时考虑。
- 可选方向包括：
  - 单独训练 `regrasp_expert`
  - 单独训练 `recovery_expert`
  - 蒸馏一个更小的 router
  - 蒸馏一个面向局部恢复的 `light expert`

## 13. 代码改造建议

### 13.1 Server 侧

- 重点文件：
  - `openpi-official/src/openpi/policies/agentic_policy.py`
- 建议新增：
  - planner 输出中的 `expert` 白名单校验
  - `planner_expert_counts`
  - `planner_latency_stats`

### 13.2 Runner 侧

- 重点文件：
  - `scripts/run_agentic_vla_libero.py`
- 建议新增：
  - `RouterDecision`
  - `ExpertState`
  - `VerifierDecision`
  - `route_expert()`
  - `verify_expert_progress()`
  - `expert_stats`
  - `current_expert`
  - `steps_in_expert`

### 13.3 建议的数据结构

- `RouterDecision`
  - `expert`
  - `trigger`
  - `reason`
  - `request_plan`
  - `subgoal`
  - `prompt_override`
  - `client_hints`
- `ExpertState`
  - `current_expert`
  - `current_subgoal`
  - `steps_in_expert`
  - `retry_count`
  - `last_trigger`
- `VerifierDecision`
  - `status`
  - `reason`

## 14. 日志与统计

建议新增并统一记录：

- `expert_counts`
- `expert_success_counts`
- `expert_escalation_counts`
- `expert_avg_steps`
- `planner_event_counts`
- `planner_action_counts`
- `planner_expert_counts`
- `planner_avg_latency_ms`
- `planner_p95_latency_ms`
- `recovery_success_after_expert_invocation`
- `effective_control_hz`

## 15. Benchmark 计划

### 主对比

- `Baseline`
  - 纯 `default_vla_expert`
- `Agentic`
  - 当前 `on-demand planner + structured JSON + FSM`
- `MoE-Agentic v1`
  - `Router + Expert + Verifier + planner-assisted routing`

### 消融

- `w/o transition_expert`
- `w/o recovery_expert`
- `w/o regrasp_expert`
- `w/o planner_assistance`
- `w/o verifier`
- `w/o critic_signal`

### 指标

- success rate
- avg episode length
- planner trigger count
- planner latency
- expert usage distribution
- success rate after expert invocation
- avg recovery steps
- effective control frequency

## 16. 真机部署考虑

- 这一版 MoE 比“planner 常驻、每步规划”更适合真机。
- 主要原因：
  - planner 低频触发
  - recovery mode 明确
  - verifier 可解释
  - 延迟预算更容易控制
  - expert 行为更容易做安全审计

### 推荐部署方式

- 执行层：`10Hz`
- 重规划层：`1Hz` 或更低
- planner：仅在需要时调用
- 动作执行：优先 `action chunk + 短时缓存 + 小步恢复`

## 17. 当前最值得立即实现的 v1

- 在 `agentic_policy.py` 中新增：
  - `expert` 输出
  - `expert` 白名单校验
  - `planner_expert_counts`
- 在 `run_agentic_vla_libero.py` 中新增：
  - `current_expert`
  - `route_expert()`
  - `verify_expert_progress()`
  - `expert_stats`
  - `VerifierDecision`
- 保持现有 `on-demand planner + structured action + subgoal FSM` 主干不变。
- 先把 `MoE` 做成“系统层专家路由版本”，再决定是否进入学习式 router 或轻量 expert 训练。

## 18. 最终建议

- 先做 `Agent-level MoE`，不要一开始就做模型内部 MoE。
- 第一阶段不要急着重训大模型。
- 优先把你当前已有创新统一为：
  - `experts`
  - `router signals`
  - `verifier rules`
- 先把系统做成：
  - 更清晰的论文叙事
  - 更完整的消融结构
  - 更可部署的具身 Agent 框架

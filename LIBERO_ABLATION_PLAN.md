# Agentic-VLA 消融实验计划 (Updated 2026-05-13)

## 实验代号与设计

| 代号 | 变体 | 验证的原创创新 | 实现方式 |
|------|------|---------------|---------|
| **A1** | pi05_libero fine-tuned (baseline) | 基线 | 无额外增强 |
| **A2** | A1 + **Transition Agent** | DAG-EAM: State Gap bridging via conditional edge | 检测停滞→发送过渡prompt→恢复任务prompt |
| **A3** | A1 + **Graph RAG + Evo-KAM Memory** | 拓扑子图检索 + 终身记忆进化 | 注入场景先验(z_offset/yaw/force)到prompt |
| **A4** | A1 + **Critic/Retry** | 反思闭环: 独立验收 + 错误归因 + 参数修正 | Qwen3-VL视觉判断→recovery→retry |
| **A2+A3** | Transition + Graph RAG | 过渡+检索联合 | 组合 |
| **A2+A4** | Transition + Critic | 过渡+反思联合 | 组合 |
| **Full** | Transition + Graph RAG + Critic | **完整五层闭环** | 全部启用 |

> 注：Vision Prompt (原B1) 不再作为核心消融项，降级为附录补充实验。
> 原因：在fine-tuned ID场景下增益≈0%，且有VLA²方法重叠问题。

## 已完成数据

| 代号 | Suite | 成功率 | 状态 |
|------|-------|--------|------|
| A1 | libero_spatial | 99.0% (198/200) | ✅ |
| A1 | libero_object | 98.0% (196/200) | ✅ |
| A1 | libero_goal | 98.0% (196/200) | ✅ |
| A1 | libero_10 | 90.0% (180/200) | ✅ (Task 8: 55% 关键弱点) |

## 实验重点

- **主要战场**: 官方 `LIBERO` benchmark 的 `libero_10` suite（长程任务，A1 Task 8仅55%，提升空间最大）
- **辅助验证**: libero_goal, libero_object (短任务上验证不降级)
- **暂缓**: libero_spatial (天花板99%，无需重复)

说明：

- `libero_10` 不是为 Agentic-VLA 专门定制的 task 集，而是 `LIBERO` 官方 benchmark 自带的标准 suite
- 我们的实验设计是：在该官方 benchmark 上定义 `A1/A2/A3/A4/Full` 消融协议，并重点分析弱点 task 的机制改进

## A2 Transition Agent 实现方案

核心思想：在VLA action chunk执行后检测State Gap，通过DAG条件边触发过渡。

1. 每个replan_steps后，检测end-effector位移变化
2. 如果连续N步位移<阈值(停滞/抖动)，判定为State Gap
3. 向policy server发送过渡prompt: "move to a safe neutral position and open gripper"
4. 执行过渡action chunk (约10-20步)
5. 恢复原始task prompt继续推理

这是**我们自己的设计**：用VLA自身的prompt-based transition，而非Sci-VLA的GPT代码生成。

## A3 Graph RAG + Memory 实现方案

1. 在第一个atomic task完成后，Memory记录执行参数
2. 下一个atomic task开始时，检索相似场景先验
3. 注入到prompt中（如 "pick up the moka pot, use z_offset 0.02"）

## A4 Critic/Retry 实现方案

1. 需要Qwen3-VL-8B做视觉判断
2. 加载策略：临时加载Qwen3-VL判断后卸载
3. 判断逻辑：截取当前观测→Qwen3-VL回答"Is the subtask completed? Is the gripper stuck?"
4. 如果失败：触发recovery→修改prompt重试
5. **独特设计**：错误归因分类(collision/slip/misgrasp)→自适应重试策略

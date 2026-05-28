# Agentic-VLA CCF-A 论文推进计划

## 1. 当前总体判断

项目方向是正确的，但要达到优秀的具身智能 CCF-A 论文标准，必须满足三条硬约束：

1. 所有结论都基于真实 LIBERO rollout，而不是估算、mock 或脚本内启发式数字。
2. 论文主贡献必须落在 `fine-tuned VLA + Agentic enhancement` 上，而不是 `base model`。
3. 需要证明 Agentic 模块带来的提升是**可重复、可归因、对长程/弱点任务有效**，而不是只在简单 ID 任务上刷更高天花板数字。

## 2. 当前最有价值的论文定位

建议把论文主线收敛为：

- 基座：`pi05_libero` fine-tuned VLA
- 问题：即使 fine-tuned VLA 在简单 suite 上接近饱和，在官方 `LIBERO` benchmark 的 `libero_10` suite 这类长程任务上仍存在显著弱点
- 方法：通过 Agentic 机制补足
  - `Transition Agent`：解决 state gap / 卡滞 / chunk 间断裂
  - `Graph RAG + Memory`：为下一子任务注入结构化场景先验
  - `Critic/Retry`：独立验收、错误归因与恢复重试
- 结论：Agentic 模块不一定提高简单任务上限，但能显著改善长程任务鲁棒性、失败恢复和弱点任务成功率

## 3. 当前最可信的主实验战场

### 主战场

- 官方 `LIBERO` benchmark 的 `libero_10` suite

原因：

- A1 baseline 已有真实弱点：整体 `90.0%`
- `Task 8 = 55%`，具有充足提升空间
- 适合证明 Transition / Critic / Memory 的真实价值

说明：

- `libero_10` 是 `LIBERO` 官方提供的标准 task suite，不是我们专门为本项目挑选的自定义 benchmark
- 我们自己的实验设计体现在：基于这一官方 benchmark 构造面向 Agentic-VLA 的消融协议、弱点任务分析和恢复统计

### 辅助验证

- `libero_object`
- `libero_goal`

原因：

- 需要证明 Agentic 增强不会显著破坏短程任务性能
- 可作为“不降级”与泛化性的辅助表

### 不建议继续作为主表重点

- `libero_spatial`

原因：

- baseline 已接近天花板
- 在主表中难以体现 Agentic 模块优势

## 4. 论文最强的主贡献表述

建议主贡献不要写成“又一个带 RAG 的机器人框架”，而应强调：

1. **针对长程 VLA 的 Agentic 增强范式**
   - 不是替代 VLA，而是为其补上过渡、结构化先验和恢复机制
2. **显式状态化的执行闭环**
   - 从单次 chunk 预测提升到可检测、可恢复、可回退的过程控制
3. **错误归因驱动的重试策略**
   - 不仅判断成功/失败，还决定如何恢复

## 5. 目前必须警惕的风险

### 风险 1：文档叙事不一致

仓库内历史文档仍混有旧阶段的描述。论文写作时必须以以下证据为准：

- `results/.../summary.json`
- rollout 视频
- 当前 `scripts/run_agentic_vla_libero.py`

### 风险 2：A3 当前实现还偏“prompt prior injection”

现阶段 `Graph RAG + Memory` 更接近可解释的 prompt 先验注入，而不是完整图检索系统。论文里需要谨慎表述：

- 可称为 `topology-aware scene priors`
- 如果没有完整向量检索与记忆更新证据，不宜过度宣称“完整 Graph RAG 系统”

### 风险 3：A4 当前 Critic 还需要更强的真实验收定义

如果 Critic 只靠 VLM 图像问答，而没有结合环境状态变化，审稿人可能质疑：

- 是否只是“额外的语言模型打分器”
- 是否存在视觉误判而非真正 grasp 成功验证

因此 A4 最好加入：

- 视觉判断
- 环境状态或轨迹变化特征
- retry 次数与收益统计

### 风险 4：实验量需要更完整

若只给单次长跑，论文说服力不够。至少需要：

- 多 suite
- 弱点 task 分析
- 重试/恢复统计
- 消融组合

## 6. 接下来实验计划

### 第一阶段：跑通主线消融

按如下优先级：

1. `A2: A1 + Transition`
2. `A3: A1 + Graph RAG + Memory`
3. `A4: A1 + Critic/Retry`
4. `A2+A3`
5. `A2+A4`
6. `Full: A2+A3+A4`

所有主实验先在 `libero_10` 上跑。

### 第二阶段：验证不降级

对以下 suite 跑：

- `libero_object`
- `libero_goal`

只需验证：

- Full 是否不明显破坏基线
- A2/A4 是否在部分弱点任务上仍有增益

### 第三阶段：补强分析

需要增加以下统计：

- 每个 episode 的 transition 次数
- Critic 检查次数 / 触发重试次数
- 重试后成功率
- 弱点 task 的失败类型分布
- episode 时长和额外计算代价

## 7. 论文实验表设计

### 主表 1：四个 suite 总体成功率

- A1
- A2
- A3
- A4
- Full

### 主表 2：`libero_10` 分任务结果

重点突出：

- Task 8
- Task 6
- Task 0/3

### 主表 3：机制统计

- 平均 transition 次数
- 平均 critic retry 次数
- retry 后恢复成功率
- 平均 episode 长度

### 图 1：弱点任务前后对比

- baseline vs Full 在 `Task 8` 上的成功率
- baseline vs Full 的失败类型变化

## 8. 论文写作计划

### 8.1 标题方向

建议标题围绕：

- Agentic enhancement for long-horizon VLA
- reflective recovery
- topology-aware priors
- embodied manipulation

### 8.2 摘要结构

摘要建议四句式：

1. VLA 在 LIBERO 上很强，但长程任务仍有脆弱性
2. 提出 Agentic-VLA，为 VLA 增加过渡、结构化先验和反思恢复
3. 在官方 `LIBERO` benchmark 上做真实 rollout 消融，重点改善 `libero_10` suite 中的长程弱点任务
4. 方法提升成功率并改善恢复与稳定性

### 8.3 方法章节

建议按模块写：

1. Baseline VLA pipeline
2. Transition Agent
3. Graph RAG + Memory priors
4. Critic/Retry loop
5. Unified execution flow

### 8.4 实验章节

建议结构：

1. Setup
2. Main Results
3. Ablation
4. Weak-task Analysis
5. Cost / Latency / Retry Analysis

## 9. 当前最推荐的推进策略

- 主表基线采用 `pi05_libero`，不要再把 `pi0_base` 放进主结果叙事
- 把 `Vision Prompt` 固定放附录
- 把 `libero_10` 作为论文主战场
- 把 `Task 8` 作为案例分析核心
- 先用真实实验把 `A2`、`A4` 做强，再决定是否把 `A3` 扩展成更完整的 Graph RAG 版本

## 10. 一句话结论

Agentic-VLA 目前已经走在正确方向上，但要达到优秀 CCF-A 论文标准，接下来的关键不是继续堆模块，而是用真实、可审计的 `libero_10` 消融结果证明：

- VLA 的长程弱点真实存在
- Agentic 机制能稳定修复这些弱点
- 这种修复是可解释且可归因的

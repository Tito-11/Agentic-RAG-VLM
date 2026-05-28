# Full Experiment Plan

## 2026-05-15 Current Objective

当前实验目标已经从“先拿到一个完整 `Full on libero_10` 主结果”切换为“在不破坏现有投稿主线的前提下，补齐 `v2` 定向修正证据并系统完成低成功率任务消融”：

- 保留当前论文主结果 `results/ablation_FULL_tuned_libero10_20260513_225055/`，最终数字为 `91/100 = 91.0%`
- 不改写主文主线：仍然只以 `A1 baseline + Full` 作为论文主结果
- 把新增 `v2` 实验定位为 targeted evidence，用于解释弱点任务、决定是否值得重跑整套 `libero_10`
- 在所有补实验里坚持真实 `official LIBERO simulator rollout`、可审计 JSON、可回溯视频，不接受估计值或 mock

## Why V2 Is Needed

- 当前 `Full-Agentic-VLA-Tuned` 虽整体高于 baseline：
  - `Full = 91.0%`
  - `A1 baseline = 90.0%`
- 但弱点任务仍集中在：
  - `Task 8 = 40%`，低于 baseline 的 `55%`
  - `Task 9 = 90%`，低于 baseline 的 `95%`
  - `Task 6 = 80%`，仅与 baseline 持平
- 重新审计 `run_agentic_vla_libero.py` 后已确认一处明确实现问题：
  - 旧版 `Scene Priors` 在复合任务中会让 `target-region` 先验覆盖 `manipulated-object` 先验
  - 对 `moka pot + stove`、`mug + microwave` 这类任务，会削弱“稳定抓取并保持竖直运输”的关键约束

## V2 Implemented Fixes

- `Scene Priors / Memory`
  - 将先验拆分为 `manipulated_object` 与 `target_region/target_container`
  - 改为组合 object prior 与 target prior，而不是简单覆盖
  - 补充 `mug`、`microwave` 等 task keyword
- `Transition Agent`
  - 对 `stove / microwave` 任务使用 task-aware transition prompt
  - 更明确强调 `keep object upright`、`re-center above burner / with opening`
- `Critic / Recovery`
  - 对 `stove / microwave` 任务使用 task-aware recovery prompt
  - 延后最早触发步数，减少过早误判
- `Task-aware control profile`
  - 对 `stove / microwave` 任务使用更保守的 `transition_min_steps / critic_min_steps`
  - 收紧 stall window 与 threshold

## Current Real Status

- 主结果目录：
  - `results/ablation_FULL_tuned_libero10_20260513_225055/`
  - 状态：已完成，可用于论文主表
- `v2` 低成功率任务矩阵已完成：
  - `Task 8`: `Full=70%`, `Transition=80%`, `Graph=60%`, `Critic=70%`
  - `Task 9`: `Full=100%`, `Transition=90%`, `Graph=100%`, `Critic=90%`
  - `Task 6`: `Full=90%`, `Transition=100%`, `Graph=80%`, `Critic=90%`
- 与旧结果对比：
  - `Task 8`: `40% -> 70%`
  - `Task 9`: `90% -> 100%`
  - `Task 6`: `80% -> 90%`
- 当前判断：
  - `Graph + Memory` 在弱点任务上有正向但任务依赖的贡献
  - `Critic` 仍未触发有效 retry，证据偏弱
  - `Task 8` 的精简方案 B 已完成：
    - `Full w/o Graph = 60%`
    - `Full w/o Critic = 70%`
    - `Full w/o Transition = 80%`
  - 这说明当前 `Task 8` 的瓶颈不是“某个模块完全无效”，而是完整堆叠后的协同仍未调到最佳

## Planned Execution Order

按优先级顺序执行：

1. 已完成：`Full v2 on Task 8`
2. 已完成：`A2 / A3 / A4 v2 on Task 8`
3. 已完成：`Full v2 on Task 9`
4. 已完成：`A2 / A3 / A4 v2 on Task 9`
5. 已完成：`Full v2 on Task 6`
6. 已完成：`A2 / A3 / A4 v2 on Task 6`
7. 已完成：对 `Task 8` 补精简版方案 B
8. 当前决定：不再继续扩张实验，进入论文收口与投稿阶段

当前统一执行脚本：

```bash
bash /home/admin1/ct/Agentic-VLA/scripts/run_low_success_ablations.sh
```

默认矩阵：

- task 顺序：`8 9 6`
- variant 顺序：`full transition graph critic`
- 每个 task/variant：`10 trials`

精简版方案 B 执行脚本：

```bash
bash /home/admin1/ct/Agentic-VLA/scripts/run_full_drop_ablations.sh
```

默认矩阵：

- task：`8`
- variant：`wo_graph wo_critic wo_transition`
- 每组：`10 trials`

## Decision Rule

- 当前已确认 `Full v2` 在 `Task 8/9/6` 上均有改善，因此无需继续补单模块矩阵
- 当前也已完成 `Task 8` 的精简方案 B，用于判断完整堆叠中的模块相互作用
- 当前结论：
  - `Graph + Memory` 不是负资产，去掉后 `Task 8` 会从 `70%` 降到 `60%`
  - `Critic` 当前基本中性，去掉后 `Task 8` 维持在 `70%`
  - `Transition` 单独强，但在当前 full-stack 中存在 interaction issue，因为 `Task 8` 上 `w/o Transition = 80%`
- 因此当前最合理的动作不是继续整套重跑，而是：
  - 保留 `91.0%` 主结果
  - 将 `Task 8/9/6` 与 `Task 8` 方案 B 作为 targeted evidence
  - 进入论文收口与投稿阶段

## Cleanup Policy

- 绝不删除任何有审计价值的真实结果目录
- 可安全清理的对象仅限：
  - 论文 LaTeX 构建中间文件：`.aux`、`.fdb_latexmk`、`.fls`、`.log`
  - 已被 `fig.1-gemini.png` 到 `fig.4-gemini.png` 替换且不再用于论文正文的旧占位图
- 若某文件仍被论文正文、报告正文或审计链路直接引用，则不得删除

## Submission Boundary

- 当前投稿前强制完成项：
  - `A1 baseline on libero_10`
  - `Full on libero_10`
  - `v2` 弱点任务 targeted evidence
- 当前投稿前非强制项：
  - `Full on libero_object`
  - `Full on libero_goal`
  - `libero_90`

## Resume Command

如需继续补齐低成功率任务矩阵：

```bash
bash /home/admin1/ct/Agentic-VLA/scripts/run_low_success_ablations.sh
```

如需继续补精简版方案 B：

```bash
bash /home/admin1/ct/Agentic-VLA/scripts/run_full_drop_ablations.sh
```

如需恢复已冻结的 `A2`：

```bash
bash /home/admin1/ct/Agentic-VLA/scripts/resume_a2_transition_libero10.sh
```

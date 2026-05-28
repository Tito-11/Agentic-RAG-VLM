# Agentic-VLA Results Evidence Guide

本文件用于说明 `results/` 目录中哪些结果是当前投稿主线，哪些结果是保留证据，哪些结果暂不作为主文使用。

## 1. 当前投稿主线

- 主基线：
  - `results/ablation_B0_pi05_libero_10_20260512/summary.json`
  - 用途：`libero_10` 主比较基线
- 当前主实验：
  - `results/ablation_FULL_refined_libero10_20260518_2059/`
  - 用途：当前论文主结果线，统一版 `Full-Agentic-VLA-Refined`（`libero_10`，每 task 20 次）
  - 关键文件：
    - `summary.json`：最终总结果
    - `_partial_summary.json`：运行中阶段性结果
    - `_completed_tasks.json`：已完成 task 列表
    - `_resume_state.json`：可恢复状态
    - `videos/`：真实 rollout 视频

## 2. 机制诊断（Task 8 的 w/o 消融）

- 目录：
  - `results/ablation_FULL_tuned_v2_wo_graph_task8_20260516_112247/`
  - `results/ablation_FULL_tuned_v2_wo_critic_task8_20260516_112247/`
  - `results/ablation_FULL_tuned_v2_wo_transition_task8_20260516_112247/`
- 用途：
  - 给出最小但关键的机制诊断证据（在 `Task 8` 上对 Full stack 的协同关系做排查）
  - 辅助讨论：`Transition` 往往是最稳定的增益来源，`Critic` 在当前部署中仍未触发有效 retry

## 3. 保留证据（不作为主线数字来源）

- 上一版主实验目录（保留）：
  - `results/ablation_FULL_tuned_libero10_20260513_225055/`
  - 用途：历史对照与复现回溯，不再作为主表数字来源

## 8. 当前最重要的保存地址

- 论文主稿：
  - `paper/Agentic-VLA/agentic_vla_paper_v1.tex`
- 当前主实验目录：
  - `results/ablation_FULL_refined_libero10_20260518_2059/`
- 主基线目录：
  - `results/ablation_B0_pi05_libero_10_20260512/`
- `Task 8` w/o 消融目录：
  - `results/ablation_FULL_tuned_v2_wo_graph_task8_20260516_112247/`
  - `results/ablation_FULL_tuned_v2_wo_critic_task8_20260516_112247/`
  - `results/ablation_FULL_tuned_v2_wo_transition_task8_20260516_112247/`

## 9. 当前最终结果状态

- 当前主实验：`results/ablation_FULL_refined_libero10_20260518_2059/`
- 当前状态：已完成
- 最终结果：`185/200 = 92.5%`
- 主基线结果：`180/200 = 90.0%`
- 当前主实验分任务结果：
  - `Task 0`: `95%`
  - `Task 1`: `100%`
  - `Task 2`: `90%`
  - `Task 3`: `95%`
  - `Task 4`: `95%`
  - `Task 5`: `100%`
  - `Task 6`: `80%`
  - `Task 7`: `100%`
  - `Task 8`: `75%`
  - `Task 9`: `95%`
- 说明：
  - 上述数字来自官方 `LIBERO` 模拟器中的真实 rollout 最终 `summary.json`。
  - 成功判定来自真实环境执行中的 `env.step(...)->done`，不是 mock，不是离线估计。
  - 当前投稿主线可以正式使用该目录中的最终结果文件回填论文主表与结果段落。
  - 结果应保守表述为：Refined Full 在 `libero_10` 上对强 baseline 取得 `+2.5` 个百分点的真实提升，且弱点任务 `Task 8` 得到显著修复，但仍存在个别非弱点任务的小幅波动。

## 10. 本次投稿的实验边界

- 本次投稿前，主结果只承诺完成：
  - `Full on libero_10`
- `libero_object`、`libero_goal` 的 `Full` 结果本次不再作为必须完成项。
- 这两组实验若后续继续推进，其定位是 non-degradation check，而不是替代 `libero_10` 主结果。

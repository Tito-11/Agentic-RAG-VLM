# Agentic-VLA

`Agentic-VLA` 是一个面向长程具身操作任务的研究型项目：在保留强 `VLA` 低层执行器的前提下，引入按需触发的高层 `VLM planner`、失败分类、恢复闭环与轻量验证机制，提升 `LIBERO` 长程任务上的鲁棒性、可解释性与部署可行性。

本项目当前聚焦的不是“再训练一个更大的动作模型”，而是验证：

- 强 `fine-tuned VLA` 是否仍需要外挂式 `Agentic` 机制
- `planner-on-demand` 是否能在资源受限条件下工作
- `Transition / Memory / Critic / Router / Verifier` 这类模块如何以可审计方式增强长程执行

## 项目亮点

- **真实评测链路**：所有主结论都来自官方 `LIBERO` 模拟器中的真实 rollout，而不是 mock、离线估计或硬编码成功率。
- **强基线设定**：使用 `pi0.5 / pi05_libero` 作为默认动作专家，而不是弱得不公平的 `base model`。
- **Agentic 增强而非替换**：在 `VLA` 之外加入 `Transition`、`GraphRAG + Memory`、`Critic/Retry`、结构化 planner 和验证闭环。
- **面向论文与部署**：强调可追溯结果、明确的失败边界、模块化代码结构与可复现实验入口。

## 当前系统

- **Low-level executor**：`pi0.5 / OpenPI`
- **High-level planner**：`Qwen3-VL-8B-Instruct`
- **Planner mode**：`on-demand`
- **Control target**：`10Hz`
- **Planner output**：结构化 JSON，而不是自由文本
- **关键机制**：
  - `failure taxonomy`: `stall / collision / misgrasp / slip`
  - `subgoal + FSM + verifier`
  - `cooldown / cache / max_plans_per_episode`
  - `Transition / Critic / GraphRAG-Memory`
  - `Router + Expert + Verifier` 的 `MoE v1` 骨架

## 已完成的核心实验

以下结果均来自真实 `LIBERO` rollout，并已在项目日志中做过审计记录。

### 强基线

- `pi05_libero` on `libero_spatial`: `99.0% (198/200)`
- `pi05_libero` on `libero_object`: `98.0% (196/200)`
- `pi05_libero` on `libero_goal`: `98.0% (196/200)`
- `pi05_libero` on `libero_10`: `90.0% (180/200)`
- 关键弱点：`libero_10 Task 8 = 55%`

### 当前主结果

- `Full-Agentic-VLA-Refined` on `libero_10`: `92.5% (185/200)`
- 相比强基线：`+2.5` 个百分点
- 关键弱点修复：
  - `Task 8`: `55% -> 75%`
- 当前观察到的主要收益来源：
  - `Transition` 是最稳定的增益模块
  - `GraphRAG + Memory` 在部分复杂任务上有帮助
  - `Critic` 当前更像检查器，而不是已被完全证明的强恢复模块

### 结果边界

- `pi0_base` 在当前 `LIBERO` 真实链路上的先导结果很弱，不能作为论文主基线。
- `Vision Prompt` 已做过接入与补充实验，但不再作为主论文创新点。
- 当前主实验结论应保守表述为：
  - `Agentic-VLA` 在强 `pi0.5` baseline 上取得了可审计但不夸张的真实增益。

## 仓库结构

```text
Agentic-VLA/
├── README.md
├── EXPERIMENT_LOG.md
├── RESULTS_EVIDENCE_GUIDE.md
├── AGENT_LEVEL_MOE_PLAN.md
├── scripts/
│   ├── run_agentic_vla_libero.py
│   ├── launch_full_libero10.sh
│   ├── prepare_libero_norm_stats.py
│   └── download_qwen3_vl.py
├── openpi/
│   ├── scripts/serve_policy.py
│   └── src/openpi/policies/agentic_policy.py
├── agentic_vla/
│   └── ...
└── paper/
    └── Agentic-VLA/
```

## 关键代码入口

- 真实 `LIBERO` 评测入口：`scripts/run_agentic_vla_libero.py`
- policy server：`openpi/scripts/serve_policy.py`
- server 侧 agentic planner 协议：`openpi/src/openpi/policies/agentic_policy.py`
- `Agent-level MoE v1` 设计文档：`AGENT_LEVEL_MOE_PLAN.md`

## 文档导航

- 结果与证据边界：`RESULTS_EVIDENCE_GUIDE.md`
- 按时间排序的真实实验日志：`EXPERIMENT_LOG.md`
- 当前 `MoE v1`/`Router + Expert + Verifier` 设计：`AGENT_LEVEL_MOE_PLAN.md`
- 论文主稿目录：`paper/Agentic-VLA/`

## 运行概览

### 1. 准备 OpenPI / LIBERO 环境

优先参考：

- `openpi/README.md`
- `openpi/examples/libero/README.md`

### 2. 启动 policy server

```bash
PYTHONPATH=/path/to/openpi/src python openpi/scripts/serve_policy.py --env LIBERO --port 8000
```

若启用 agentic planner：

```bash
PYTHONPATH=/path/to/openpi/src python openpi/scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  --agentic \
  --planner-model /path/to/Qwen3-VL-8B-Instruct
```

### 3. 运行 LIBERO 评测

```bash
python scripts/run_agentic_vla_libero.py \
  --task-suite libero_10 \
  --trials 20 \
  --host 127.0.0.1 \
  --port 8000 \
  --transition \
  --graph-rag \
  --critic \
  --agentic-planner
```

## 当前研究定位

- **主 benchmark**：`LIBERO`, 尤其是 `libero_10`
- **主 baseline**：`pi0.5 / pi05_libero`
- **主目标**：
  - 资源受限下的 `Agentic` 框架设计
  - `VLA` 推理优化与实时性统计
  - benchmark 与后续真机部署可行性

## 说明

- 仓库中仍保留部分原型代码和论文资产，用于研究复现与后续扩展。
- 历史结果目录、权重和本地运行缓存默认不纳入 Git 跟踪。
- 若你希望基于本仓库复现论文级结论，优先以 `scripts/` 与 `openpi/` 下的真实评测链路为准。

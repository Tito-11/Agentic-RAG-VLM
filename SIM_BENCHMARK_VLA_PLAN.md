# 仿真 VLA 实验规划（Isaac Sim / Isaac Lab(Orbit) + LIBERO / RobotWin 等）

## 0. 目标与原则

目标：在仿真环境（Isaac Sim + Orbit/Isaac Lab 等）与标准 benchmark（LIBERO、RobotWin 等）上，系统化运行 VLA（openpi policy server + websocket client）实验，并形成可审计、可复现、可扩展的评测管线。

原则：

- **统一评测口径**：固定 task suite、trials、seed、成功判定规则；输出统一 `summary.json`，可直接生成论文表格与图。
- **解耦策略与仿真**：策略推理（policy server）与仿真环境（client runner）分离，使用 websocket 通信，避免依赖地狱。
- **可审计**：保存 `summary.json / _partial_summary.json / _resume_state.json` 与必要视频（主结果与弱任务），可追溯到每条 episode。
- **可扩展**：新增 benchmark 只需实现一个适配器（observation/action/success 接口）和 task list 配置。

## 1. 统一系统架构

### 1.1 双进程（推荐）

- 进程 A（GPU 1）：policy server（openpi-official）
  - `scripts/serve_policy.py --env <LIBERO|DROID|...>`，端口默认 `8000`
- 进程 B（GPU 0 或同卡）：sim/benchmark runner（本仓库）
  - 负责：reset/step、渲染多相机图像、构造 observation payload、接收 action chunk、记录视频和结果

好处：仿真栈（Isaac/Orbit）与策略栈（JAX/torch/量化 Qwen critic 等）互不污染，便于部署和复现。

### 1.2 统一 Observation/Action 协议（对齐 openpi 推理接口）

参照当前 `LIBERO` runner 的 payload 结构：

- Observation：
  - `observation/image`：主视角 RGB（uint8, HxWx3，resize+pad 到 224）
  - `observation/wrist_image`：腕部相机 RGB（可选但推荐）
  - `observation/state`：连续状态向量（EEF pos + axis-angle + gripper qpos；或按 checkpoint 需要扩展）
- Command：
  - `prompt`：自然语言任务描述（可选加入 task priors / memory / refinement）
- Action（policy server 返回）：
  - `actions`：长度为 N 的 action chunk 列表；runner 每次取前 `replan_steps` 执行

关键：不同 benchmark 可能需要不同 checkpoint（例如 `pi05_libero` vs `pi05_droid`），但 payload 的键名与数值类型要保持一致或按 server metadata 进行兼容适配。

## 2. Benchmark 分层路线图

### 2.1 第一层：已跑通的 LIBERO（作为黄金参考）

目标：把 LIBERO 的 runner 作为“golden harness”，把后续仿真 benchmark 的实现都对齐这个输出与审计习惯。

交付物：

- 复用现有入口：`scripts/run_agentic_vla_libero.py`
- 统一结果格式：`summary.json` + `_resume_state.json` + 视频
- 论文主表口径：`libero_10, 20 trials/task`

### 2.2 第二层：Isaac Sim / Isaac Lab(Orbit) 通用 VLA Runner

目标：在 Orbit/Isaac Lab 上跑“可控、可重复”的 manipulation tasks，并能接入 openpi policy server。

需要做的关键工作：

1) **环境适配器（Environment Adapter）**
   - 实现 `openpi_client.runtime.environment.Environment` 的 4 个方法：
     - `reset()`
     - `is_episode_complete()`
     - `get_observation()`：返回原始观测（至少两路相机 + 机器人状态）
     - `apply_action(action: dict)`：将 VLA action 转换为仿真控制（EEF delta pose / joint control / gripper）

2) **动作映射（Action Mapping）**
   - 明确 VLA action 的语义与尺度：
     - 位置/姿态 delta 的单位（米/弧度）
     - 控制频率与仿真步长
     - gripper open/close 的阈值或连续控制
   - 提供“safe scaling profile”，避免因尺度不匹配导致不稳定。

3) **成功判定（Success Signal）**
   - 尽量使用 benchmark 官方的 success（例如 Orbit task 直接给 `success` flag）
   - 若只能启发式判定：必须记录判定逻辑并在论文中保守表述

4) **渲染与录像**
   - headless 渲染（优先 egl）
   - 固定相机位姿与分辨率
   - 每个 episode 输出 mp4（主表可只留抽样，弱任务全留）

建议的最小落地目标（MVP）：

- 先实现 3–5 个 Orbit 任务（pick&place / drawer / door / stack 等）
- 每 task 先跑 `5 trials` 做 smoke，再扩到 `20 trials`

### 2.3 第三层：RobotWin 等额外 benchmark

目标：对 RobotWin 采用同样的适配方式，快速形成可比较的结果矩阵。

落地步骤：

- 读 RobotWin 官方接口：
  - 任务定义、reset、success 判定、动作空间（EEF vs joint）、相机接口
- 写 `RobotWinAdapter`：
  - 输出 `observation/image` 与 `observation/state`
  - 兼容 `actions` 的执行频率
- 先跑 baseline（不加任何 agentic 模块）建立零样本下界，再跑你的增强模块

## 3. 实验矩阵建议（期刊导向）

### 3.1 每个 benchmark 的最小必跑

- Baseline（A1）：纯 VLA（对应 benchmark 合适 checkpoint）
- Full（你的方法）：Transition / Memory / Critic 等（如适用）
- 指标：
  - success rate（主指标）
  - average episode length / steps
  - transition count / success when used
  - retry count / recovery success（如果 critic 真触发）
  - 运行时开销（FPS 或 step latency）

### 3.2 Trials 与复现策略

- 主表对标口径：**20 trials/task**
- 稳定性增强（可选）：
  - 对最关键弱任务补到 `50 trials/task`，或
  - 全套再跑一个不同 seed（第二次 run）

## 4. 工程实现计划（可直接落地到仓库）

### Phase A（1–2 天）：Orbit/Isaac Lab MVP 打通

- 新增 `scripts/run_vla_sim_benchmark.py`（通用 runner）
- 新增 `sim_envs/orbit_env.py`（Orbit 适配器）
- 新增 `sim_tasks/orbit_task_suites.py`（task suite 配置）
- 输出结果格式完全对齐 `results/<run_name>/summary.json`

验收标准：

- 能在 headless 模式跑完 1 个任务的 3 个 trials 并生成视频与 summary

### Phase B（2–4 天）：RobotWin 适配

- 新增 `sim_envs/robotwin_env.py`
- 新增 `sim_tasks/robotwin_task_suites.py`
- 先跑 baseline，再跑 Full

### Phase C（投稿级）：结果表与审计链路

- 统一生成表格脚本（沿用当前 `paper/Agentic-VLA/export_results_tables.py` 的风格）
- 统一机制统计输出（transition/retry 等）
- 弱任务失败模式与视频抽样（用于定性图）

## 5. 风险与对策

- **Domain mismatch**：`pi05_libero` 对 Isaac/RobotWin 可能零样本很差
  - 对策：先跑 baseline 下界；必要时换更通用 checkpoint（如 droid 系列）或做轻量微调
- **Action scaling 不匹配**：导致抖动或撞击
  - 对策：增加 action scaling profile；先用小尺度 + rate limit；保留 safety reset
- **成功判定不统一**：审稿人质疑公平性
  - 对策：优先使用官方 success；启发式必须写清并提供视频与统计
- **渲染/显存冲突**：Isaac 与推理模型抢 GPU
  - 对策：server/client 分卡；或 server 单卡、仿真 headless 降分辨率；必要时远端 server

## 6. 你接下来立刻可以做的事（最短路径）

1) 先选 Orbit 的 3–5 个任务作为 `orbit_5` 套件
2) 用 `pi05_libero` 先跑 baseline 5 trials/task（验证链路）
3) 再跑你的 Full 模块 5 trials/task（验证是否有趋势）
4) 趋势成立后，扩到 20 trials/task，并写入论文补充实验

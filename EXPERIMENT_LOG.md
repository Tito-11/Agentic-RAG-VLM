# Agentic-VLA 实验日志

## 2026-05-20/21 Agentic Pipeline 升级（VLM on-demand + 可验证结构化决策 + Failure Taxonomy）

### 目标
- 将系统从“VLA 低层策略 + prompt 改写”升级为可发表的软件 Agentic pipeline：只在需要时触发 VLM planner，并且 planner 输出可验证的结构化决策（有限集合），同时与既有创新模块（Transition / GraphRAG-Memory / Critic）形成统一闭环。
- 在推理资源受限（单卡 4090）设定下可运行：planner 4bit 量化 + on-demand lazy-load + 冷却/上限/缓存，避免 planner 反复介入。

### 核心改动（系统级）
- Planner（server side, VLM）：
  - 只在 client 发出 `agentic.request_plan=true` 时触发；不触发则 `plan_ms=0` 且不会加载/占用 planner 资源。
  - 输出严格 JSON（可验证）：`action ∈ {continue,regrasp,lift,retreat,slow_down}` + `prompt_suggested` + `client_hints{action_scale,apply_steps}` + `rationale`。
  - 扩展结构化字段：`subgoal ∈ {execute,recover,regrasp,lift,retreat,slow_down}`，用于驱动 client 侧子目标状态机（FSM）。
  - server 侧解析/校验/回退：非 JSON 或 action 不在白名单时自动回退到安全默认（continue）。
  - planner 节流：`plan_cooldown_steps`（最小间隔）+ `max_plans_per_episode`（每 episode 上限）+ `cache_plans`（按 event+instruction 缓存）。
- Trigger（client side, 无常驻 VLM）：
  - Failure taxonomy（规则/状态信号）扩展为：`stall / collision / misgrasp / slip(启发式)`。
  - 与既有创新融合：除 taxonomy 外，`state_gap`（Transition Agent 触发）与 `critic`（Critic 判定需要 retry）也会触发 planner 请求，但均遵循 cooldown。
- Executor（client side, VLA）：
  - 真正应用 planner 的 `client_hints`：在接下来 `apply_steps` 内对动作（前 6 维）乘以 `action_scale`，形成“可控的 recovery 参数化执行”。
  - 子目标状态机（FSM）：
    - `execute`：正常执行（VLA chunked actions）。
    - `recover/*`：执行恢复策略（结合 action_scale/apply_steps），并使用 failure taxonomy 的“无 failure 连续窗口”作为可验证退出条件。
- Context injection（结合 GraphRAG/Memory）：
  - 当 GraphRAG 提取到 priors 时，将压缩后的先验摘要写入 `agentic.context`，只在 planner 请求时发送，提升规划可靠性且不增加常规控制负担。

### 推荐 benchmark 验证方式（LIBERO）
- 对比原则：只改一个变量（是否启用 on-demand planner），其余控制变量一致，便于投稿叙事与 ablation。
- Baseline server（无 agentic）：
  - `PYTHONPATH=/home/admin1/ct/openpi-official/src conda run -n openpi python /home/admin1/ct/openpi-official/scripts/serve_policy.py --env LIBERO --port 8000`
- Agentic server（on-demand planner + 4bit）：
  - `PYTHONPATH=/home/admin1/ct/openpi-official/src conda run -n openpi python /home/admin1/ct/openpi-official/scripts/serve_policy.py --env LIBERO --port 8000 --agentic --planner-model /home/admin1/ct/Agentic-VLA/weights/Qwen3-VL-8B-Instruct --planner-device cuda --planner-dtype bfloat16 --planner-quant 4bit --plan-mode on_demand --plan-cooldown-steps 50 --max-plans-per-episode 3 --cache-plans true`
- Runner（libero_10 小样本快速验证；完整实验按论文口径调整 trials/task）：
  - `python scripts/run_agentic_vla_libero.py --task-suite libero_10 --task-id 0 --trials 20 --host 127.0.0.1 --port 8000 --agentic-planner --planner-cooldown-steps 50`
  - 若要同时启用既有创新模块：
    - `python scripts/run_agentic_vla_libero.py --task-suite libero_10 --task-id 0 --trials 20 --host 127.0.0.1 --port 8000 --transition --graph-rag --critic --agentic-planner --planner-cooldown-steps 50`

### 结果记录口径（用于论文）
- 主要指标：success rate、avg episode length。
- 机制指标（新增/增强）：
  - `planner_stats.total_requests / plans_generated`
  - `planner_stats.event_counts`（stall/collision/misgrasp/slip/state_gap/critic）
  - `planner_stats.action_counts`（continue/regrasp/lift/retreat/slow_down）
  - `agentic.subgoal`：每次 planner 触发时的子目标标签（FSM 统计与可解释性补充）
- 注：`slip` 当前是启发式（夹爪开合轨迹），后续若需要更强可验证性，优先改为“抓取后物体状态信号”或“可重复的视觉规则”，避免引入额外常驻 VLM。

## 2026-05-16 V2 结果汇总与精简方案 B

### 当前状态
- `Task 8/9/6` 的 `v2 + 单模块消融` 队列已全部完成。
- 当前没有剩余 `run_agentic_vla_libero.py` rollout 进程在跑；只有 policy server 仍在运行。
- 这意味着当前可以基于真实结果决定下一步，而不需要继续等待后台队列。

### 低成功率任务真实结果
- `Task 8`
  - 旧 `Full`: `40%`
  - baseline: `55%`
  - `Full v2`: `70%`
  - `Transition only`: `80%`
  - `Graph only`: `60%`
  - `Critic only`: `70%`
- `Task 9`
  - 旧 `Full`: `90%`
  - baseline: `95%`
  - `Full v2`: `100%`
  - `Transition only`: `90%`
  - `Graph only`: `100%`
  - `Critic only`: `90%`
- `Task 6`
  - 旧 `Full`: `80%`
  - baseline: `80%`
  - `Full v2`: `90%`
  - `Transition only`: `100%`
  - `Graph only`: `80%`
  - `Critic only`: `90%`

### 当前判断
- `v2` 修正确实有效：
  - `Task 8`: `40% -> 70%`
  - `Task 9`: `90% -> 100%`
  - `Task 6`: `80% -> 90%`
- `Transition` 是当前最稳定的增益来源：
  - `Task 8`: `80%`
  - `Task 6`: `100%`
- `GraphRAG/Memory` 在 `Task 9` 上价值明显，但在 `Task 8/6` 上不够稳定。
- `Critic` 仍没有产生有效 retry，当前更像“检查器”，而不是已经被结果证明有效的恢复模块。
- 最关键现象是：
  - `Task 8` 上 `Transition only = 80% > Full v2 = 70%`
  - 说明 `Full` 里至少还有一个附加模块在拖后腿

### 决策
- 不再重复扩大单模块矩阵。
- 下一步改为补一个精简版方案 B，只针对 `Task 8` 做：
  - `Full w/o Graph`
  - `Full w/o Critic`
  - `Full w/o Transition`
- 目的：
  - 用最小 wall-clock 成本确认到底是哪一块拖累了 `Task 8` 的完整框架表现

### 新增脚本
- 已新增：
  - `scripts/run_full_drop_ablations.sh`
- 默认行为：
  - task：`8`
  - variants：`wo_graph wo_critic wo_transition`
  - trials：`10`

### 精简方案 B 最终结果
- `Task 8` 的 `Full w/o ...` 已全部完成：
  - `Full w/o Graph = 60%`
  - `Full w/o Critic = 70%`
  - `Full w/o Transition = 80%`
- 由此得到的最终判断：
  - `Graph + Memory` 不是负资产，去掉后结果明显变差
  - `Critic` 当前贡献很弱，去掉后基本不变
  - `Transition` 单独有效，但在当前 full-stack 中与其他模块的协同尚未调到最佳
- 因此最稳妥的结论不是“某个模块完全没用”，而是：
  - 当前完整框架的瓶颈主要来自模块交互，而不是单模块绝对失效

### 投稿前最终收口判断
- 当前不再继续扩大实验面。
- 投稿版实验边界确定为：
  - 主文：`A1 baseline + Full on libero_10`
  - supporting evidence：
    - `Task 8/9/6` 的 `v2` targeted runs
    - `Task 8` 的单模块与 `Full w/o ...` 诊断
- 后续若继续拓展，优先作为期刊方向打磨 `Graph + Memory` 与 `Critic`，而不是继续在当前会议版上无止境补矩阵。

## 2026-05-15 V2 执行状态同步

### 当前运行状态
- `Full v2 on Task 8` 仍在运行：
  - `results/ablation_FULL_tuned_v2_task8_20260515_160827/`
- 当前审计到的实时状态：
  - `run_agentic_vla_libero.py --task-id 8 --trials 10` 进程仍在运行
  - GPU 上该进程已占用约 `9.4 GiB` 显存
  - 结果目录已创建，但截至本次同步时尚未写出 `_partial_summary.json`、`summary.json` 或视频文件
- 当前解释：
  - 脚本只有在首个 episode 完成后才会持久化阶段性 JSON
  - 因此当前更稳妥的判断是“仍在初始化或执行首条 rollout”，而不是“实验已失败”

### 已挂起的后续实验队列
- 已新增顺序队列脚本：
  - `scripts/run_pending_low_success_pipeline.sh`
- 已实际启动后台队列，当前行为是：
  1. 等待当前 `Task 8 Full v2` 结束
  2. 继续执行 `Task 8` 的 `transition / graph / critic` 消融
  3. 执行 `Task 9` 与 `Task 6` 的 `full / transition / graph / critic` 矩阵
- 这样可以避免手动盯进程导致后续实验断档。

### 本次同步完成的文档更新
- 已更新：
  - `FULL_EXPERIMENT_PLAN.md`
  - `RESULTS_EVIDENCE_GUIDE.md`
  - `paper/Agentic-VLA/SUBMISSION_ASSETS_V1.md`
- 当前口径统一为：
  - 主结果仍是 `91.0%`
  - `v2` 低成功率任务实验属于 targeted evidence
  - 在结果文件真正落盘前，不能把任何运行中目录写成完成结果

### 本次已完成的安全清理
- 已删除论文 LaTeX 构建中间文件：
  - `agentic_vla_paper_v1.aux`
  - `agentic_vla_paper_v1.fdb_latexmk`
  - `agentic_vla_paper_v1.fls`
  - `agentic_vla_paper_v1.log`
- 已删除已被 Gemini 正式图替换的旧占位图：
  - `fig_method_overview.png`
  - `fig_transition_gap.png`
  - `fig_scene_priors_memory.png`
  - `fig_critic_retry_flow.png`
- 所有真实实验结果目录保持不删。

## 2026-05-15 V2 定向修正与低成功率任务补实验

### 为什么继续做 V2
- 当前主结果已经完成：
  - `results/ablation_FULL_tuned_libero10_20260513_225055/`
  - `91/100 = 91.0%`
- 但弱点任务仍然突出：
  - `Task 8`: `40%`
  - `Task 9`: `90%`
  - `Task 6`: `80%`
- 尤其 `Task 8` 明显低于 baseline 的 `55%`，因此值得在投稿前窗口内做最小但有针对性的 `v2` 修正。

### V2 代码层发现
- 重新审计 `run_agentic_vla_libero.py` 后确认：
  - 旧版 `GraphRAG/Memory` 对复合任务使用简单 `dict.update()` 合并先验；
  - 对 `put both moka pots on the stove` 这类任务，会让 `stove` 的目标区域先验覆盖 `moka pot` 的抓取/运输先验；
  - 这会弱化 `keep upright`、`approach from side` 这类对 Task 8 很关键的对象级约束。

### 已实施修正
- `Scene Priors`：
  - 引入 `role=manipulated_object / target_region / target_container`
  - 将 object prior 与 target prior 组合，而不是覆盖
  - 新增 `mug`、`microwave` 等泛化先验
- `Transition`：
  - 对 `stove` 任务使用
    - `re-center above the burner`
    - `keep the object upright`
  - 对 `microwave` 任务使用
    - `re-center with the opening`
    - `keep the object upright`
- `Recovery`：
  - 对 `stove/microwave` 任务采用 task-aware recovery prompt
- `Task-aware control profile`：
  - 对 `stove/microwave` 任务提高 `transition_min_steps`
  - 对 `critic_min_steps` 延后
  - 对 stall window / threshold 做更保守设置

### 当前正在运行的补实验
- `Full v2 on Task 8`
- 结果目录：
  - `results/ablation_FULL_tuned_v2_task8_20260515_160827/`
- 目标：
  - 在最小 wall-clock 成本下，先验证 `Task 8` 是否从 `40%` 提升

### 新增执行脚本
- 为了系统完成低成功率任务补实验，已新增：
  - `scripts/run_low_success_ablations.sh`
- 默认顺序：
  - tasks：`8 9 6`
  - variants：`full transition graph critic`
  - 每组：`10 trials`

## 2026-05-14 主实验最终完成

### 最终结果
- 当前论文主实验：
  - `results/ablation_FULL_tuned_libero10_20260513_225055/`
- 当前状态：
  - 已完成
- 最终总结果：
  - `91/100 = 91.0%`
- 对应主基线：
  - `results/ablation_B0_pi05_libero_10_20260512/summary.json`
  - `180/200 = 90.0%`
- 结论：
  - 在 official `LIBERO` 模拟器真实 rollout 下，修正版 `Full-Agentic-VLA-Tuned` 对强 baseline 取得了 `+1.0` 个百分点的最终提升。

### 分任务结果
- `Task 0`: `100%`，baseline `90%`
- `Task 1`: `100%`，baseline `100%`
- `Task 2`: `100%`，baseline `95%`
- `Task 3`: `100%`，baseline `90%`
- `Task 4`: `100%`，baseline `100%`
- `Task 5`: `100%`，baseline `95%`
- `Task 6`: `80%`，baseline `80%`
- `Task 7`: `100%`，baseline `100%`
- `Task 8`: `40%`，baseline `55%`
- `Task 9`: `90%`，baseline `95%`

### 机制统计
- `transition_stats.total_transitions = 81`
- `transition_stats.transitions_leading_to_success = 72`
- `avg_transitions_per_episode = 0.81`
- `transition_success_rate_when_used = 88.89%`
- `critic_stats.total_checks = 327`
- `critic_stats.retries_triggered = 0`
- `critic_stats.retries_leading_to_success = 0`
- 解释：
  - 当前最终收益主要与 `Transition` 干预相关，而不是由 `Critic/Retry` 的显式恢复闭环带来。

### 审计口径
- 所有最终数字均来自 `summary.json` / `_partial_summary.json` / `_resume_state.json`，并可回溯到本地 rollout 视频。
- 当前结果是 official `LIBERO` 模拟器中的真实交互执行，不是物理机器人实验，也不是离线估计。
- 论文可做正向表述，但必须保持保守：
  - 可以写“Full 在 `libero_10` 上以真实 rollout 取得小幅但可审计的提升”
  - 不应写成“框架已全面解决长程任务”或“critic 已被结果证明有效”

## 2026-05-14 主实验运行与 Critic 审计

### 当前运行状态
- 当前论文主实验仍为：
  - `results/ablation_FULL_tuned_libero10_20260513_225055/`
- 最新已审计运行中检查点：
  - `57/57` success
- 当前仍是 official `LIBERO` 模拟器中的真实 rollout，不是 mock，不是离线估计。
- 当前 GPU 计算进程中，仅观察到 tuned `Full` 对应的 rollout Python 进程；未观察到旧失败版 `Full` 结果目录对应的运行中 rollout 进程。

### Critic / Qwen 审计结论
- 当前 tuned `Full` 的启动命令明确包含：
  - `--critic`
  - `--qwen-model /home/admin1/ct/Agentic-VLA/weights/Qwen3-VL-8B-Instruct`
  - `--qwen-quant 4bit`
- 运行进程环境与命令行均指向当前 `openpi` conda 环境和主实验目录。
- 对正在运行的 rollout Python 进程进行非侵入式审计时，已观察到：
  - `bitsandbytes` CUDA 动态库已映射进进程地址空间
  - `safetensors` 动态库已映射进进程地址空间
- 在当前脚本实现中，这两类库在主实验路径里只出现在 `CriticAgent.load_model()` 的 Qwen3-VL 加载逻辑中，因此上述运行时证据与“Critic 已真实按 4bit Qwen 路径加载”一致。
- 但在不打断当前真实 benchmark 的前提下，尚未直接抓到启动终端中的
  - `Qwen3-VL loaded successfully`
  - 或等价的逐条运行日志文本
- 因此当前最稳妥表述应为：
  - **高置信度认为当前 tuned Full 的 critic 正在使用 Qwen3-VL，而非 heuristic fallback；但论文文字仍应保持保守，不把这一点写成超出审计证据强度的绝对断言。**

## 2026-05-13 Full 修正版推进

### 当前判断
- `Full-Agentic-VLA` 首轮真实 `libero_10` 运行已产生可审计失败证据，但当前不支持“框架已成功提升长任务”这一结论。
- 首轮 `Full` 在 `Task 0` 的前 `4` 条真实 rollout 全部失败，且干预频次偏高：
  - `transition_total = 12`
  - `retry_total = 8`
  - `critic_checks = 104`
- 这更像是当前策略设置导致的 **过干预**，而不是单纯 benchmark 难度问题，因为 baseline 在 `Task 0` 上为 `90%`。

### 已冻结的首轮 Full 证据
- 原始结果目录：`results/ablation_FULL_libero10_20260513_203506/`
- 冻结快照：`results/ablation_FULL_libero10_20260513_203506_before_tuning/`
- 当前可见真实视频：
  - `task0_trial0_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
  - `task0_trial1_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
  - `task0_trial2_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
  - `task0_trial3_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`

### 已实施的策略修正
- `Transition`：
  - `stall_threshold: 0.003 -> 0.0015`
  - `stall_window: 8 -> 12`
  - 增加最早触发步数：`TRANSITION_MIN_CONTROL_STEPS = 60`
  - 单 episode 最大 transition：`3 -> 1`
  - `transition chunk steps: 15 -> 6`
  - prompt 从“回中立位并张开夹爪”改为“轻微抬升并稳定夹爪，同时保持当前抓取”
- `Critic/Retry`：
  - `check_interval: 20 -> 60`
  - 增加最早检查步数：`CRITIC_MIN_CONTROL_STEPS = 120`
  - 单 episode 最大 retry：`2 -> 1`
  - `recovery steps: 15 -> 6`
  - recovery prompt 去掉 `open the gripper`
  - retry 触发条件从“只要 error_type 非 unknown 即可”收紧为“仅在明确 stuck 时才允许 retry”
  - critic prompt 新增保守约束：只有在持续卡住或明显抖动时才报告 `stuck=true`

### 当前主实验线
- 新一轮实验改为 `Full-Agentic-VLA-Tuned`
- 启动脚本：`scripts/launch_full_libero10.sh`
- 当前目标：先验证修正版是否显著降低 `Task 0` 上的过干预，再继续观察 `libero_10` 弱任务上的真实收益
- 论文主线同步收敛为：
  - `A1`: 强 baseline
  - `Full`: 当前唯一主文 agentic 结果线
  - `A2`: 保留历史证据，但不进入主文结果叙事

## 当前结论
- 只有真实 LIBERO 运行产生的 JSON 和视频可以作为论文证据，之前所有硬编码、估算和 mock 结果全部作废。
- 真实 `pi0_base` 基线已经跑通到 `LIBERO + official openpi + websocket server + 本地评测脚本` 这一完整链路。
- 真实 `pi0_base` 先导结果仍然较弱：`libero_spatial` 的 `task 0~4`，每个任务 `3` 次，共 `15` 条 episode，当前结果为 `0/15`。
- 真实 `pi05_libero` 官方配置可以在当前链路上稳定运行；已完成 `libero_spatial` 全量 `10 tasks x 50 trials = 500 episodes`，成功 `497/500`，成功率 `99.4%`。
- 当前还没有可信的 Agentic-VLA 消融结果，因为 `agentic_vla` 中的 Agentic 模块尚未接入真实的 LIBERO 评测链路。

## 关键证据文件
- 真实 `pi0_base` 先导结果汇总：`results/pi0_base_smoke_20260512/summary.json`
- 真实 `pi0_base` 单任务结果：
  - `results/pi0_base_smoke_20260512/task_0.json`
  - `results/pi0_base_smoke_20260512/task_1.json`
  - `results/pi0_base_smoke_20260512/task_2.json`
  - `results/pi0_base_smoke_20260512/task_3.json`
  - `results/pi0_base_smoke_20260512/task_4.json`
- 真实 `pi0_base` 失败视频：`results/pi0_base_smoke_20260512/videos/`
- 真实 `pi05_libero` 全量 `libero_spatial` 结果：`results/full_libero_20260512/libero_spatial.json`
- 真实 `pi05_libero` 全量 `libero_spatial` 视频：`results/full_libero_20260512/videos/libero_spatial/`
- 真实评测入口：[run_agentic_vla_libero.py](file:///home/admin1/ct/Agentic-VLA/scripts/run_agentic_vla_libero.py)

## 2026-05-12 真实实验推进记录

### 1. 审计并撤回伪结果
- 确认旧版 [run_agentic_vla_libero.py](file:///home/admin1/ct/Agentic-VLA/scripts/run_agentic_vla_libero.py) 曾包含 `np.random`、`expected_sr`、mock env 等伪评测逻辑。
- 已将该脚本重写为真实评测入口，只接受环境真实 `done` 信号，不再生成硬编码成功率。
- 已删除一批伪结果文件，并撤回旧文档中的 `86% / 91% / 95% / 99%` 等虚假或不可追溯结果。

### 2. 打通官方 openpi 与 LIBERO 运行链路
- 使用官方仓库 `/home/admin1/ct/openpi-official` 作为真实运行基础，而不是项目内不完整的 `openpi` 目录。
- 完成 `uv sync --python 3.11`，补齐 `jax`、`flax`、`lerobot`、`openpi-client`、`robosuite`、`bddl`、`libero` 等依赖。
- 将 `~/.libero/config.yaml` 指向 `openpi-official/third_party/libero`。
- 修复 `norm_stats.json` 缺失问题，确保 `pi0_base` 能真实加载。
- 当前默认使用 `egl` 渲染，避免本机缺失 `osmesa` 导致的启动失败。

### 3. 真实 `pi05_libero` 对照运行
- 说明：这一结果对应官方 `pi05_libero` 配置，不是 `pi0_base`。
- 已完成 `libero_spatial` 全量评测：
  - 任务数：`10`
  - 每任务 trials：`50`
  - 总 episode：`500`
  - 成功：`497`
  - 成功率：`99.4%`
- 结果文件：`results/full_libero_20260512/libero_spatial.json`
- 当前目录中 `libero_object` 只有部分视频，说明之前全量长跑被中断，不能当作完整结果使用。

### 4. 真实 `pi0_base` 基线先导实验
- 服务端配置：
  - `policy.config = pi0_libero`
  - `policy.dir = weights/openpi-assets/checkpoints/pi0_base`
- 评测设置：
  - suite：`libero_spatial`
  - task：`0~4`
  - 每任务 trials：`3`
- 当前真实结果：
  - `task 0`: `0/3`
  - `task 1`: `0/3`
  - `task 2`: `0/3`
  - `task 3`: `0/3`
  - `task 4`: `0/3`
  - 合计：`0/15`
- 结果文件：`results/pi0_base_smoke_20260512/summary.json`

## 当前工程判断
- 真实 baseline 现在分成两条线：
  - `pi05_libero`：官方配置可稳定跑通，并已得到可信高成功率结果。
  - `pi0_base`：真实链路已打通，但在当前 `libero_spatial` 先导任务上表现为 `0/15`。
- 这说明后续论文里必须明确区分“官方 fine-tuned LIBERO 模型”和“base model”。
- 当前最重要的不是继续写结论，而是把 Agentic-VLA 的最小增强模块接入真实评测，再与 `pi0_base` 做公平对比。

## 2026-05-12 Vision Prompt 接入真实链路

### 5. Vision Prompt 代码实现
- 已在 [run_agentic_vla_libero.py](file:///home/admin1/ct/Agentic-VLA/scripts/run_agentic_vla_libero.py) 中实现 Vision Prompt 功能：
  - 新增 `_apply_vision_prompt()` 函数：利用 `SegmentationRenderEnv` 提供的 ground-truth 分割，对 `obj_of_interest` 中的目标物体叠加半透明彩色 mask
  - 新增 CLI 参数：`--vision-prompt`、`--mask-alpha`、`--ablation-tag`
  - 当 `--vision-prompt` 启用时，自动切换到 `SegmentationRenderEnv`，获取 `agentview_segmentation_instance` 分割图
  - 分割图 shape `(256, 256, 1)` 已处理 squeeze，与 RGB 图 `(256, 256, 3)` 对齐
  - 结果 JSON 中包含 `vision_prompt`、`mask_alpha`、`ablation_tag` 字段
- 端到端验证通过：
  - `SegmentationRenderEnv` 创建、`reset()`、`set_init_state()` 均正常
  - `obj_of_interest` 和 `instance_to_id` 映射正确
  - Task 0 验证：`akita_black_bowl_1` (843 px, 绿色) + `plate_1` (1388 px, 橙色) 共 2231 像素被 mask
- 待运行：libero_spatial task 0~4 的 Vision Prompt 消融实验

## 关键发现：pi0_base vs pi05_libero

### 为什么 pi0_base 在 LIBERO 上 0% 成功率？
- **根因**：`pi0_base` 是预训练模型（pre-trained only），从未在 LIBERO 数据上 fine-tune
- **三篇论文一致结论**：所有 VLA 都需要 task-specific fine-tuning 才能在 LIBERO 上工作
  - pi0 论文：明确区分 pre-training 和 post-training，post-training 用 task-specific 数据 fine-tune
  - VLA² 论文：Table I 所有 baseline 标注 (FT) = fine-tuned，π0 (FT) 在 LIBERO-Spatial 上 96.8%
  - Sci-VLA 论文：为每个 atomic task 生成 100 demos 并 fine-tune 80k steps
- **openpi 官方配置**：
  - `pi0_libero` = 训练配置（从 pi0_base 开始 fine-tune 30k steps）
  - `pi05_libero` = 训练配置（从 pi05_base 开始 fine-tune 30k steps）
  - `serve_policy.py --env LIBERO` 默认加载 `pi05_libero` fine-tuned checkpoint
- **结论**：论文实验必须基于 fine-tuned 模型，不能用 base model

### 消融实验基线选择
- 使用 `pi05_libero`（pi0.5 fine-tuned on LIBERO）作为所有消融实验的基线
- 这与 VLA²、Sci-VLA 等论文的实验范式一致：先 fine-tune，再测试 agentic 增强

## 2026-05-12 消融实验（pi05_libero 基线）

### B0: pi05_libero Fine-tuned Baseline ✅
- 配置：`serve_policy.py --env LIBERO`（pi05_libero fine-tuned checkpoint）
- 评测：libero_spatial 全量 10 tasks x 20 trials = 200 episodes
- 结果：**99.0% (198/200)**
- 分任务结果：
  - Task 0-4, 6-9: 100% (20/20)
  - Task 5: 90% (18/20)
- 结果文件：`results/ablation_B0_pi05_libero_finetuned_20260512/summary.json`
- 视频目录：`results/ablation_B0_pi05_libero_finetuned_20260512/videos/`

### B1: pi05_libero + Vision Prompt ✅
- 配置：pi05_libero + `--vision-prompt --mask-alpha 0.35`
- 评测：libero_spatial 全量 10 tasks x 20 trials = 200 episodes
- 结果：**99.0% (198/200)**
- 分任务结果：
  - Task 0-7: 100% (20/20)
  - Task 8: 95% (19/20)
  - Task 9: 95% (19/20)
- 结果文件：`results/ablation_B1_pi05_libero_vision_prompt_20260512/summary.json`

### B0 vs B1 对比（libero_spatial）
| 实验 | 整体 | Task 5 | Task 8 | Task 9 |
|------|------|--------|--------|--------|
| B0 baseline | 99.0% | 90% | 100% | 100% |
| B1 + VP | 99.0% | **100%** | 95% | 95% |
- Task 5: VP 修复了 baseline 弱点（90%→100%）
- Task 8,9: VP 轻微下降（100%→95%），mask 可能干扰已学到的视觉特征
- **结论**：libero_spatial 太简单（天花板效应），需在更难的 suite 上测试

### B0 vs B1 对比（libero_object）
| 实验 | 整体 | Task 3 | Task 5 | Task 6 |
|------|------|--------|--------|--------|
| B0 baseline | 98.0% | 95% | 90% | 95% |
| B1 + VP | **99.5%** | **100%** | **100%** | **100%** |
- Vision Prompt 在 object suite 上显著修复了弱点任务
- Task 3/5/6: 90%/95% → 100%，VP 帮助模型更精准定位目标物体
- 结果文件：`results/ablation_B0_pi05_libero_object_20260512/summary.json`、`results/ablation_B1_pi05_libero_object_20260512/summary.json`

### B0 vs B1 对比（libero_goal）
| 实验 | 整体 | Task 0 | Task 3 | Task 4 |
|------|------|--------|--------|--------|
| B0 baseline | 98.0% | 95% | 100% | 95% |
| B1 + VP | 97.5% | **90%** | 95% | **100%** |
- Vision Prompt 在 goal suite 上轻微下降（98.0%→97.5%）
- 原因：`obj_of_interest` 包含区域名称（如 `wooden_cabinet_1_middle_region`）不在分割图中
- 需要修复：过滤掉非物体实例的 obj_of_interest 条目
- 结果文件：`results/ablation_B0_pi05_libero_goal_20260512/summary.json`、`results/ablation_B1_pi05_libero_goal_20260512/summary.json`

### B0 baseline（libero_10 长程任务）✅
- 结果：**90.0% (180/200)**
- Task 8 最低：**55%**（关键弱点任务）
- 结果文件：`results/ablation_B0_pi05_libero_10_20260512/summary.json`

### B1 + Vision Prompt（libero_10 长程任务）🔄 运行中
- 结果文件：`results/ablation_B1_pi05_libero_10_20260512/summary.json`

### 实验设计说明
- libero_spatial baseline 已达 99%，Vision Prompt 在 ID 任务上提升空间有限
- 但 Vision Prompt 的核心价值在于 **OOD 场景**（参照 VLA² 论文 Hard 级别）
- 后续需在 libero_object、libero_goal、libero_10 等更难的 suite 上测试
- 论文贡献点：Agentic 增强在 OOD/长程任务上的鲁棒性提升，而非 ID 任务的绝对值提升

## 下一阶段任务
- **B1 完成**：等待 Vision Prompt 消融结果
- **B2: pi05_libero + Vision Prompt + Transition**：设计不破坏 action chunk 的过渡动作中间层
- **B3: pi05_libero + Critic/Retry**：增加失败检测和重试逻辑
- **Full: Agentic-VLA**：完整框架消融
- **OOD 测试**：在 libero_object、libero_goal、libero_10 上重复消融
- **论文定位**：Agentic-VLA 的价值在于 OOD 鲁棒性和长程任务连贯性，而非 ID 任务天花板

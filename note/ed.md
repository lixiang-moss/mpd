# 主题：关于 Diffusion 运动规划项目（MPD）中 DPM-Solver 移植后的实验场景配置与评估策略咨询

> 背景：你在现有 MPD 项目中对比 `DDIM` 与你移植的 `DPM-Solver / DPM-Solver++`（重点关注 DPM‑Solver++ 2nd order, multistep）。你计划在 `/scripts/inference/cfgs` 下的 6 个环境上做：参数调优 → 基准对比 → 消融实验。

下面的建议结合了论文（`2412.19948v3.pdf`，对应文本 `2412.19948v3.txt`）和当前代码实现（尤其是 `env_id_replace / selection_start_goal / seed / n_start_goal_states` 的实际含义与副作用）。

---

## 0. 先把论文口径与代码口径对齐（很重要）

### 0.1 论文里 “training environment” vs “additional objects” 的含义

论文在 Fig.6 的说明里明确区分两列：

- **Training environment**：无新增障碍（“without additional objects”）。
- **Additional objects**：在场景里加入**新的障碍物**（论文用它来回答泛化/适应性问题：方法是否能在训练未见过的新障碍下仍然成功）。

论文同时强调：在 **additional objects** 情况下，所有方法的 success/validity 都会下降，尤其是“只用 prior、不做 cost 优化”的方法会掉得更厉害；而带 cost optimization（例如 MPD）的提升在 additional objects 下更明显。

> 这意味着：**“additional objects” 在论文语义中更像 OOD（out-of-distribution）测试**，不是用来调参的“训练/验证同分布场景”。

### 0.2 论文的统计方式：100 contexts × 100 trajectories

论文在 “General Results in Simulation” 段落和 Fig.6 描述里使用：

- **sample 100 contexts**（100 个 start/goal 任务实例）
- **optimize 100 trajectories per context**（每个任务实例生成/优化 100 条候选轨迹）
- 最后报告指标的 **mean/std**（标准差通常是跨 contexts 统计）

### 0.3 你的代码里这些参数分别控制什么

- `env_id_replace`（在 YAML）：覆盖数据集原始环境，换成另一个环境类（通常就是“加了额外障碍”的版本）。代码位置：`mpd-splines-public/mpd/utils/loaders.py` 里通过 `env_id_replace if env_id_replace else dataset_args["env_id"]` 选择环境类。
- `selection_start_goal`（在 `scripts/inference/inference.py`）：决定 start/goal 从哪里来：
  - `"training"`：从 train subset 抽；
  - `"validation"`：从 val/test subset 抽；
  - 其他字符串：当作一个 YAML 路径，直接读取你预先写好的 contexts 列表（`EvaluationSamplesGenerator` 支持）。
- `seed`：会同时 seed `random / numpy / torch / torch.cuda`，因此会同时影响：
  1) 抽到的 contexts（本质是 val/train 子集索引的随机排列）；  
  2) 每个 context 内 diffusion 采样的随机性（你采样 100 条轨迹的随机噪声）。
- `n_start_goal_states`：本次评测要跑多少个 context（多少个 start/goal）。
- `n_trajectory_samples`（YAML）：每个 context 内采样多少条轨迹（你现在默认 100）。

### 0.4 一个容易忽略但会影响“公平对比”的点：extra objects 会触发 context 被“拒绝并跳过”

`EvaluationSamplesGenerator.get_data_sample()` 会检查 start/goal 是否在当前环境中 valid；如果碰撞就会递归地取下一个样本。

因此：
- 在 `env_id_replace=null`（训练环境）下有效的一些 start/goal，到了 `ExtraObjects` 环境可能会变成无效；
- 这会导致 **最终真正评测到的那 100 个 contexts 在两个环境之间并不完全相同**（哪怕你固定了 seed）。

这不一定违背论文（论文也没要求 training env 与 additional objects 的 contexts 必须逐一对应），但它会影响你解释“为什么掉得多”时的严谨性。后面我会给你一个不改代码也能解决“固定 contexts” 的做法（用 `selection_start_goal` 指向一个 contexts 文件）。

---

## 1) 问题 1：`env_id_replace` 该选 `null` 还是 `ExtraObjects`？（按三个阶段分别建议）

你对 `null` vs `ExtraObjects` 的理解是对的：
- `null`：使用数据集原始环境（论文里对应 training environment）。
- `*ExtraObjects*`：新增障碍的环境（论文里对应 additional objects，用来测泛化/适应）。

### 阶段 A：参数调优（建议：主调 `null`，辅看 `ExtraObjects`）

如果你希望实验叙事尽量贴近论文（“不在 OOD 测试上调参”）：
- **主调参环境：`env_id_replace: null`**  
  理由：这更接近“模型训练分布”，能先把 solver 的数值稳定性、速度-精度折中调到合理区间。
- **每轮调参后，用固定的一小批 contexts 在 `ExtraObjects` 上做 sanity check**  
  理由：你会发现某些 solver/guidance 设定在训练环境很强，但一到新障碍就崩（validity 掉到很低）。你可以在不“用它做目标函数”的前提下，用它做“约束/筛选”。

如果你更偏工程目标（你就是要“这个场景加障碍也要最好”），那也可以直接在 `ExtraObjects` 上调参，但要在论文写作里说清楚：你调的是“目标场景最优”，而不是“泛化能力测试”。

### 阶段 B：基准对比（建议：两种环境都做，并保持同一套调好的 solver 超参）

为了让对比有说服力，建议你最终至少输出两组结果：
- `env_id_replace: null`（训练环境对比：DDIM vs DPM‑Solver++）
- `env_id_replace: <ExtraObjects>`（新增障碍环境对比：DDIM vs DPM‑Solver++）

并且**不要在这两种环境上用两套不同的 solver 超参**（否则你是在比较“各自调到最优的两个系统”，很难把差异归因到 sampler）。

### 阶段 C：消融实验（建议：至少把 `ExtraObjects` 作为重点展示场景）

论文里明确指出 additional objects 下差异更明显、也更能体现 cost optimization / guidance 的价值。所以你做 solver 变体消融（order、method、dpmsolver vs dpmsolver++）时：
- **优先在 `ExtraObjects` 上做**（最能拉开差距）
- 其次再补 `null`（说明在训练环境也不吃亏）

---

## 2) 问题 2：调参阶段要不要 6 个场景全跑？最终/消融阶段是否必须全覆盖？

### 阶段 A：参数调优（不建议一上来就 6 个全跑）

原因很现实：
- DPM‑Solver 的 steps/order/method 等组合一多，6 场景全跑会把迭代速度拖到不可用；
- 你还遇到了 Panda 3D 场景在 guidance/Jacobian 下的 OOM（这会进一步降低“快速试错”的效率）。

建议一个“从快到稳”的方案：

1) **用 1 个 Hard case 做主调参场景**：推荐 `EnvSpheres3D-RobotPanda + ExtraObjects`  
   论文里在 Fig.8 明确说它是他们考虑的最难任务（narrow passages），很适合做“把 solver 稳定性/鲁棒性逼出来”的主场景。
2) **再用 1 个便宜的 2D/Planar 场景做 sanity check**：例如 `EnvNarrowPassageDense2D` 或 `EnvPlanar4Link`  
   目的是排查“某个 solver 设定只在 Panda 上好使，别的场景反而退化/变慢”的情况。

等你把参数范围收敛后，再进入阶段 B 的全量覆盖。

### 阶段 B：最终基准对比（尽量覆盖 6 个；算力不足至少覆盖 3 类）

如果你希望“像论文一样有说服力”，最好覆盖全部 6 个（因为论文就是跨多任务展示泛化）。

但如果算力/时间实在不够，至少保证覆盖三种代表性类别：
- 2D 点质量（简单 + narrow passage）
- Planar（2-link 或 4-link）
- Panda（Spheres3D + Warehouse）

这样读者能看到：低维/中维/高维都成立。

### 阶段 C：消融实验（不一定必须 6 个全覆盖，但要选“最能拉开差距”的场景）

消融的目的不是“再证明一次全任务有效”，而是证明你选择的 DPM‑Solver++(2nd, multistep) 的优越性。

建议优先选：
- `EnvSpheres3D-RobotPanda + ExtraObjects`（最难、差异最大）
- 再加一个低维任务（比如 `EnvNarrowPassageDense2D`）证明不是只在 Panda 上成立

如果你还有算力，再把 6 个补齐会更漂亮，但不是必须。

---

## 3) 问题 3：如何设置 `seeds` 与 `n_start_goal_states` 才有统计意义？调参 vs 最终出图要区分吗？

### 先给一个核心结论（和你前面讨论一致）

> **同一组对比实验里，“更推荐把 `n_start_goal_states` 设大”，而不是用很多 seed 且每次只跑 1 个 context。**

理由：
- 论文的统计方差主要来自 **contexts 的多样性**（不同 start/goal 难度差异很大），因此“多 contexts”比“多 seed 但单 context”更能降低方差；
- 多 seed 且每次 `n_start_goal_states=1` 会把“抽题”和“采样噪声”都混在一起，解释起来更困难，也更浪费启动/预热开销。

### 阶段 A：参数调优（快为主）

推荐：
- `n_start_goal_states = 5 ~ 20`（够用来稳定排序，但仍然快）
- `seeds = 1 ~ 3 个固定 seed`（例如 0/1/2）  
  用来检查“这个配置是否偶然赢”。

并且强烈建议：
- 同一轮 sweep 里，所有配置使用**同一个 seed**（或同一组 seed），保证“同题同考”。

### 阶段 B：最终出图（对齐论文口径）

如果你要严格对齐论文：
- `n_start_goal_states = 100`
- 每个 context：`n_trajectory_samples = 100`
- seed 可以固定 1 个（论文没要求跨 seed 报告；论文的 std/mean 是跨 contexts）

如果你想更“统计学严谨”（代价是算力）：
- 固定同一批 contexts（见下一节的 `selection_start_goal` 文件方式）
- 再跑 `seeds = 3`（0/1/2）  
  这样你能区分：
  - 跨 contexts 的方差（任务难度）
  - 跨 seed 的方差（采样随机性）

### 阶段 C：消融实验（介于两者之间）

常见做法：
- `n_start_goal_states = 50`（比调参更稳，比最终更省）
- `seeds = 1 ~ 3`（看你算力）

---

## 4) 问题 4：`selection_start_goal` 应该怎么设？设了 `"validation"` 后问题 1 还重要吗？

### 4.1 是否都应该用 `"validation"`？

**最终评测/出图：建议都用 `"validation"`**（或论文语义里的 test set）。理由：
- 它对应“没见过的 contexts”，更符合论文评测；
- 用 training set 更容易高估效果，也更难解释。

调参阶段：
- 可以短期用 `"training"` 做 debug（跑得更快/更容易看到趋势），但最终一定要回到 `"validation"` 复核。

### 4.2 已经用 `"validation"` 了，`env_id_replace` 就无所谓吗？

**不是的，依然非常重要。**

原因：`selection_start_goal` 决定“start/goal 从数据集哪部分来”，但 `env_id_replace` 决定：
- 碰撞/有效性判定是在**哪个环境**里做；
- cost guidance 的梯度是在**哪个环境**里算；
- 因此最终 `fraction_valid / success / collision_intensity` 等都会变。

另外，正如 0.4 所说：在 `ExtraObjects` 环境下，一部分 validation contexts 可能会被拒绝并跳过，导致最终 contexts 集合发生偏移；所以它不仅影响指标计算，还会间接影响“你到底评测了哪些题”。

### 4.3 你观察到“在 null 上调参 → ExtraObjects 效果很差”是否等于过拟合？

不一定是“过拟合”，更可能是论文预期的现象：
- `ExtraObjects` 本质是新增障碍的 OOD 场景；
- 在论文里，很多方法在 additional objects 下 success/validity 都会明显下降；
- 能否在 additional objects 下恢复性能，关键依赖 **cost optimization / guidance 是否足够强**，以及 sampler 是否能稳定承载这类强 guidance。

如果你的目标是“既要训练环境好，又要新增障碍也好”，更合理的做法是：
- 仍然在 `null` 上做主调参（避免把测试场景当训练目标）；
- 但在选择最终超参时，把 `ExtraObjects` 作为一个“鲁棒性约束”（比如要求 `fraction_valid` 不低于某阈值），而不是完全不看。

### 4.4 强烈推荐：用 `selection_start_goal` 指向一个 “contexts 文件” 来锁定评测集（可显著提升公平性）

你现在的对比很容易被“抽到哪些 contexts”影响。最稳的做法是：
- 先固定生成一份 contexts YAML（比如 100 个），然后：
  - 所有 sampler/超参配置都用同一份文件作为 `selection_start_goal`
  - 这样不同配置就是严格意义的“同题同考”

项目代码已经支持：`selection_start_goal` 传入一个 YAML 路径时，会从该文件读取 `q_pos_start/q_pos_goal/ee_pose_goal`（`EvaluationSamplesGenerator` 里 `select_start_goal_from_file` 分支）。

**注意**：如果你希望 “training env vs extra objects” 使用同一批 contexts，则需要确保这批 contexts 在 extra objects 环境中 start/goal 也有效；否则会触发拒绝逻辑导致集合偏移。实践里更常见的做法是：
- training env 用一份 contexts 文件
- extra objects 用另一份 contexts 文件（各自保证有效）
- 把它们当作两个评测集分别报告（也更接近论文的呈现方式）

---

## 5) 给你一个可落地的推荐“实验路线图”（把上面四问串起来）

### A. 参数调优（目标：找到稳定、快速、质量高的 DPM‑Solver++ 配置）
- 主场景：`EnvSpheres3D-RobotPanda`（先 `null`，再用 `ExtraObjects` 做 sanity check）
- `selection_start_goal="validation"`
- `n_start_goal_states=10~20`
- `seeds=1~3`（固定小集合）
- 为避免 Panda OOM：先把 `n_trajectory_samples` 降到 20/32；找到候选配置后再升回 100 做复核。

### B. 基准对比（目标：DDIM baseline vs 你最优的 DPM‑Solver++）
- 覆盖：尽量 6 场景；至少覆盖 2D/Planar/Panda 三类
- 两套环境都做：`null`（training env）与 `ExtraObjects`（additional objects）
- `selection_start_goal="validation"`（最好换成 contexts 文件保证严格一致）
- `n_start_goal_states=100`，`n_trajectory_samples=100`
- 固定 seed（或 3 seeds 增强严谨性）

### C. 消融实验（目标：证明你选择的 solver 变体最优）
- 重点环境：`ExtraObjects`（差异更明显）
- 场景选择：至少 1 个 hard（Panda spheres extra）+ 1 个简单（2D/planar）
- `n_start_goal_states=50~100`，`seeds=1~3`
- 对比维度建议最少包含：order=1 vs 2、method=multistep vs singlestep、dpmsolver vs dpmsolver++（以及你认为关键的 guidance 参数是否需要随 solver 变体一起调）

---

## 6) 基于现有 `logs/` 的“候选最优”配置（回答：现有数据够不够、null 下最优是谁）

> 先说结论：你当前 `logs/` 里的 sweep **足够用来给出“阶段 A 的候选最优配置”**，但**不足以给出“最终论文级别的确定最优”**。  
> 原因：多数 sweep 的每个 `run_tag` 只评了 `seeds={0,1,2}` 且 `n_start_goal_states=1`，所以每个配置只有 **3 个 contexts**（1 seed 对 1 个 context）。这能筛出稳定趋势，但仍建议用 `n_start_goal_states=20/50/100` 做最终确认。

### 6.1 `env_id_replace: null`（training environment）下，现有 logs 里最好的几个 DPM‑Solver++ 配置

下面这些都满足你强调的版本：`dpm_solver_pp` + `order=2` + `multistep`（且是在 `env_id_replace=None` 的 sweep 里筛出来的）。

1) **当前 logs 中综合最强（按 `fraction_valid_min` 排序）**  
   - `steps=44, ct=1, skip=logSNR, solver_type=dpmsolver, denoise_to_zero=1, lower_order_final=0`  
   - `pw=1.2, guide_lr=0.012, n_guide_steps=1, max_perturb_x=0.1, t_start_guide_steps_fraction=0.1`  
   - 指标（跨 seeds=0/1/2 的 3 个 contexts）：`fraction_valid_mean≈0.987, fraction_valid_min=0.97, t_inference_total_mean≈1.076s`  
   - 来源：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_pw_to2_steps44/plots/aggregated_by_run_tag.csv`

2) **次优但更“省时”（min 降一点、t 更低）**  
   - `steps=44, ct=1, multistep, logSNR, dpmsolver, dz=1, lof=1`  
   - `pw=1.1, guide_lr=0.015, n_guide_steps=1, mpx=0.1, tsg=0.1`  
   - 指标：`fraction_valid_mean≈0.977, fraction_valid_min=0.95, t_inference_total_mean≈1.004s`  
   - 来源：`mpd-splines-public/scripts/inference/logs/sweep_solver_mpd_overnight_v1/plots/aggregated_by_run_tag.csv`

3) **同一 sweep 内的稳健备选（与 #1 同一组 steps44 扫 pw）**  
   - `steps=44, ct=1, multistep, logSNR, dpmsolver, dz=1, lof=0`  
   - `pw=1.1, guide_lr=0.012, n_guide_steps=1, mpx=0.1, tsg=0.1`  
   - 指标：`fraction_valid_mean≈0.973, fraction_valid_min=0.95, t_inference_total_mean≈1.119s`  
   - 来源：同 `sweep_dpm_solver_pw_to2_steps44`

> 说明：你提到的 “null 下 pw=1.2~1.4 很好” 与现有 sweep 是一致的，但在这组 steps44 固定的 sweep 里，`pw=1.2` 是最稳的点，`pw≥1.6` 开始出现明显不稳定（`fraction_valid_min` 断崖式下降）。

### 6.2 `env_id_replace: EnvSpheres3DExtraObjectsV00`（additional objects）下，现有 logs 的最优“鲁棒”区域长什么样

在你现有、且包含 `seeds=0/1/2` 的 replace sweep 中，表现最好的区域大致是：
- `pw≈1.1`（而不是 1.2/1.4）  
- 需要更“强/更早”的引导：更大的 `t_start_guide_steps_fraction`（例如 0.3），以及更多 solver steps（例如 84/100）  

例如现有 logs 中（按 `fraction_valid_min` 排序）：
- `steps=100, pw=1.1, guide_lr=0.01, tsg=0.3, mpx=0.15, dz=1, lof=1`  
- 指标：`fraction_valid_mean≈0.527, fraction_valid_min≈0.28`（跨 3 个 seed contexts）  
- 来源：`mpd-splines-public/scripts/inference/logs/sweep_replace_wide_steps_lr_tsg_v1/plots/aggregated_by_run_tag.csv`

> 这能解释你观察到的现象：**同一个 “pw 偏大” 的配置在 null 很好，但在 ExtraObjects 会明显掉**——本质是 prior 和环境分布错配时，过强的 prior 会抵消/压制 cost guidance 的纠偏。

### 6.3 关于 `pw` 与 `guide_lr` 的“联动”：必须二选一吗？

不一定必须二选一，取决于你要做的实验口径：

- **按论文口径做泛化评测（推荐写作主线）**：  
  在 `null` 上调参（得到 #1/#2 这种配置），然后**不改超参**直接上 `ExtraObjects` 报结果；ExtraObjects 掉分是“泛化难度”的一部分。

- **如果你想要一个“单配置兼顾两种环境”的工程最优**：  
  就必须做折中（multi-objective），例如把目标改成 “在 null 表现高的同时，ExtraObjects 的 `fraction_valid_min` 不要太低”。这通常会把 `pw` 往 1.0~1.1 拉，并通过增大 `tsg/steps` 或适度调高 `guide_lr` 来补回性能。

- **如果你允许“分场景两套最优配置”**：  
  也可以（null 用 `pw≈1.2`，ExtraObjects 用 `pw≈1.1` + 更强引导），但写作里要明确你是在做 “per-scenario tuning”，且 baseline（DDIM）理论上也应允许同样的 per-scenario tuning 才完全公平。

---

## 7) 你要的“完整指标数据”：不只看 `fraction_valid`

你说得对：挑超参不能只看 `fraction_valid`。你现在的 logs 里其实已经记录了很多指标（平滑度、末端误差、碰撞强度、路径长度、时间），只是我前面没把它们完整展开。

下面所有数值都来自对应 sweep 的 `sweep_metrics.csv`，并且我按你当前常用设置 **`seeds={0,1,2}`、`n_start_goal_states=1`** 汇总，所以每个配置只有 **3 个 contexts**：我用 **mean ± std（跨 3 个 seed contexts）**，并额外给出 **min/max**（方便你看“最差 case”）。

> 提醒：如果某个 seed 对应的 context 上 `fraction_valid=0`，那么一些“只对 valid 轨迹定义”的指标（如 `path_length_valid_mean`）在该 seed 会是 `nan`，此时 std 也会变得更“漂”。这也是为什么论文要用 100 contexts。

### 7.1 现有 logs 中「null 下最强」配置（#1）在 null 与 ExtraObjects 的完整指标

**配置（你前面问的 “综合最强” 那个）**  
`run_tag = dpm_solver_pp_steps44_order2_ct1_mmultistep_skiplogSNR_soldpmsolver_lof0_dz1_pw1p2_ng1_glr0p012_mpx0p1_tsg0p1`

来源：
- null：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_pw_to2_steps44/sweep_metrics.csv`
- ExtraObjects：`mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_pw_to2_steps44_replace/sweep_metrics.csv`

**EnvA（null / training env）**（3 contexts）
- `fraction_valid = 0.987 ± 0.012`（min=0.970, max=1.000）
- `collision_intensity = 0.00128 ± 0.00154`（min=0.00000, max=0.00344）
- `ee_pos_err_best = 0.00603 ± 0.00089`
- `ee_ori_err_best = 1.763 ± 0.694`
- `ee_pos_err_mean_valid = 0.02045 ± 0.00390`
- `ee_ori_err_mean_valid = 2.377 ± 0.331`
- `path_length_best = 5.879 ± 0.862`
- `path_length_valid_mean = 5.905 ± 0.683`
- `smoothness_best = 46.35 ± 8.05`
- `t_inference_total = 1.076 ± 0.014`（`t_generator = 1.072 ± 0.014`, `t_guide = 0.746 ± 0.007`）

**EnvReplace（ExtraObjects / additional objects）**（3 contexts）
- `fraction_valid = 0.250 ± 0.271`（min=0.020, max=0.630）
- `collision_intensity = 0.122 ± 0.057`（min=0.0617, max=0.199）
- `ee_pos_err_best = 0.0188 ± 0.0044`
- `ee_ori_err_best = 2.902 ± 0.608`
- `ee_pos_err_mean_valid = 0.0270 ± 0.0017`
- `ee_ori_err_mean_valid = 4.249 ± 0.787`
- `path_length_best = 5.407 ± 1.341`
- `path_length_valid_mean = 5.257 ± 1.442`
- `smoothness_best = 42.99 ± 15.24`
- `t_inference_total = 1.133 ± 0.012`（`t_generator = 1.129 ± 0.012`, `t_guide = 0.791 ± 0.003`）

> 这就回答了你问的：“**null 下最强配置**，在 ExtraObjects 场景到底怎么样？”——从现有数据看：它在 null 几乎满分，但在 ExtraObjects 上 `fraction_valid` 明显掉（且碰撞强度、末端误差也变差）。

### 7.2 你提到的 `pw` “二选一”问题：用现有 logs 给出定量证据（固定其他参数，只扫 pw）

你说的现象在 logs 里是能直接量化出来的。对同一套固定 solver/guidance（steps44、guide_lr≈0.012、tsg≈0.1 这一组），只改 `pw`，统计结果如下（取 `pw∈[1.0,1.2]`）：

来源：`mpd-splines-public/scripts/inference/logs/compare_prior_weight_pw_replace_vs_A/aggregated_by_pw.csv`

| 环境 | pw | fraction_valid_mean | fraction_valid_min | collision_intensity_mean | smoothness_best_mean | path_length_best_mean | t_inference_total_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| EnvA（null） | 1.01 | 0.877 | 0.810 | 0.00531 | 47.12 | 6.33 | 1.067 |
| EnvA（null） | 1.10 | 0.973 | 0.950 | 0.00169 | 54.65 | 7.00 | 1.119 |
| EnvA（null） | 1.20 | 0.987 | 0.970 | 0.00128 | 46.35 | 5.88 | 1.076 |
| EnvReplace（ExtraObjects） | 1.00 | 0.310 | 0.090 | 0.14000 | 53.52 | 6.45 | 1.124 |
| EnvReplace（ExtraObjects） | 1.10 | 0.307 | 0.070 | 0.13865 | 65.87 | 6.85 | 1.121 |
| EnvReplace（ExtraObjects） | 1.20 | 0.250 | 0.020 | 0.12206 | 42.99 | 5.41 | 1.133 |

> 注：在这组对比里，EnvA（null）没有正好 `pw=1.00` 的一行，我用同组 sweep 中最接近的 `pw=1.01` 来近似对齐区间 `[1.0, 1.2]`。

**你要的结论（只基于这组固定超参）：**
- 在 **null** 下：`pw=1.2` 明显更强（`fraction_valid_min=0.97`）。
- 在 **ExtraObjects** 下：`pw` 变大反而更容易“崩”（`pw=1.2` 的 `fraction_valid_min` 跌到 0.02），最佳更靠近 `pw≈1.0~1.1`。

这说明你说的“必须权衡”是事实；但这并不代表必须“二选一”——因为你还可以通过 **提高 steps / 提前引导（tsg）/ 调整 guide_lr** 来缓解 ExtraObjects 的退化（你后面的 wide sweep 也印证了这一点）。

### 7.3 如果你强制要求“一套参数同时在 null 与 ExtraObjects 都能用”：现有 logs 里 pw∈[1,1.2] 的最佳折中是谁？

你强调“不可能分开两套配置”，那就只能做 **multi-objective**：在保证 null 不差的前提下，尽量把 ExtraObjects 拉起来。

我用你现有 logs 里**同时在两种环境都跑过**的配置做交集筛选（同一个 `run_tag`，两边都有数据），并在 `pw∈[1.0,1.2]` 内按 **ExtraObjects 的 `fraction_valid_min` 优先**排序，得到目前最好的折中点是：

**推荐折中配置（现有 logs 下最强的 “single-config-for-both”）**  
`run_tag = dpm_solver_pp_steps44_order2_ct1_mmultistep_skiplogSNR_soldpmsolver_lof1_dz1_pw1p1_ng1_glr0p02_mpx0p1_tsg0p2`

来源：
- null：`mpd-splines-public/scripts/inference/logs/sweep_solver_mpd_overnight_v1/sweep_metrics.csv`
- ExtraObjects：`mpd-splines-public/scripts/inference/logs/sweep_replace_pwscan_v1/sweep_metrics.csv`

**EnvA（null）**（3 contexts）
- `fraction_valid = 0.967 ± 0.040`（min=0.910, max=1.000）
- `collision_intensity = 0.00203 ± 0.00260`
- `ee_pos_err_best = 0.00393 ± 0.00102`
- `ee_ori_err_best = 1.128 ± 0.174`
- `ee_pos_err_mean_valid = 0.01970 ± 0.00281`
- `ee_ori_err_mean_valid = 1.925 ± 0.346`
- `path_length_best = 6.277 ± 0.983`
- `path_length_valid_mean = 6.353 ± 0.642`
- `smoothness_best = 45.62 ± 5.38`
- `t_inference_total = 1.410 ± 0.012`（`t_generator = 1.407 ± 0.012`, `t_guide = 1.093 ± 0.003`）

**EnvReplace（ExtraObjects）**（3 contexts）
- `fraction_valid = 0.427 ± 0.370`（min=0.150, max=0.950）
- `collision_intensity = 0.0701 ± 0.0565`
- `ee_pos_err_best = 0.00702 ± 0.00309`
- `ee_ori_err_best = 2.219 ± 0.869`
- `ee_pos_err_mean_valid = 0.02970 ± 0.00698`
- `ee_ori_err_mean_valid = 3.547 ± 0.192`
- `path_length_best = 6.599 ± 2.327`
- `path_length_valid_mean = 5.800 ± 1.536`
- `smoothness_best = 68.32 ± 40.78`
- `t_inference_total = 1.598 ± 0.005`（`t_generator = 1.594 ± 0.005`, `t_guide = 1.264 ± 0.005`）

> 你会看到：这个折中配置在 null 明显不如 “#1（pw=1.2, lof0）” 极致，但它把 ExtraObjects 的 `fraction_valid_min` 从 ~0.02（#1）拉到了 0.15，并且碰撞强度也显著下降（0.122 → 0.070）。

**如果你还想继续把 ExtraObjects 拉高，但仍坚持单配置**：从你现有 logs 的趋势看，你需要把搜索重点放在：
- `steps` 往 60/84/100 走（ExtraObjects 对 steps 很敏感）
- `tsg` 往 0.2~0.4 走（更早/更久的引导）
- `pw` 固定在 1.05~1.15（不要追求 null 的 1.2/1.4 那种极值）
- `guide_lr` 做一维扫描（例如 0.01/0.015/0.02/0.03），因为它确实和 `pw` 有耦合

---

如果你愿意，我可以基于你“single-config-for-both”的约束，把下一轮最有效的 sweep 网格（steps×tsg×pw×guide_lr）列成一个最省算力、最容易收敛的实验矩阵（并把你现在 logs 里的空白区域补齐），方便你后续直接跑出论文级别的数据（100 contexts）。

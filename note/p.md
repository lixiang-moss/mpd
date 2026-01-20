# 本科毕业论文 Proposal（草案）

## 一、题目
在 MPD 扩散运动规划中比较不同采样器：DDIM 与 DPM-Solver++ 的速度–质量权衡

## 二、摘要（中文）
扩散模型在机器人运动规划中展现出生成高质量、多模态轨迹的潜力，但其迭代去噪带来的推理延迟仍是实际部署的瓶颈。MPD（Motion Planning Diffusion）通常采用 DDIM 以减少采样步数；同时，DPM-Solver++、UniPC 等 ODE/高阶求解类采样器在生成式建模领域被广泛用于进一步降低步数。当前在“固定 MPD 规划管线与固定任务设置”下，DDIM 与这些更高阶采样器的速度–质量对比仍缺少系统、可复现的实验结论。

本论文在不改变 MPD 的模型权重、引导项与规划管线的前提下，仅替换采样器：对比 MPD 默认 DDIM 与移植的 DPM-Solver++（可选加入 UniPC/DDPM 作为参考），在原项目提供的 6 个测试场景上，分别于训练环境（`env_id_replace: null`）与新增障碍（`*ExtraObjects*`，对应论文 additional objects 的 OOD 测试）两类设置下，系统改变采样步数（各采样器对应的 steps/timesteps 参数）并测量成功率/可行率、轨迹质量与端到端推理时间，从而刻画速度–质量权衡并给出在不同实时约束下的采样器与步数选择建议。本工作定位为本科毕业论文，重点在工程复现、对照实验设计与可复现结论。

## 三、引言与动机
在存在障碍物与约束的规划问题中，同一 start–goal 往往存在多条可行路径，解的分布具有多模态特征。扩散模型（如 DDPM）能够学习并表示这类分布；MPD 将扩散先验与代价/引导（cost guidance）结合，可在新场景中生成可行、无碰撞的运动轨迹。然而扩散规划通常需要多次去噪迭代，推理速度相较经典规划器（如 RRTConnect/RRT*、CHOMP 等）偏慢。对本科项目而言，清晰地回答“在不改模型/不改任务的前提下，仅替换采样器能否更快、是否会牺牲成功率与轨迹质量、这种取舍在不同难度场景下是否一致”，既具有研究意义，也具有工程落地价值。

## 四、问题陈述（Problem Statement）
在保持 MPD 规划框架、引导项、环境与任务定义不变的条件下，仅更换采样器并在多个采样步数设置下评测：不同采样器（DDIM vs DPM-Solver++/UniPC）会如何影响规划成功率、轨迹质量与端到端推理时间？在训练环境与新增障碍（OOD）两类设置下，这种差异是否一致？是否能给出随场景难度变化的“推荐采样器/步数”规律？

## 五、研究问题（Research Questions）
- RQ1（核心对比）：在相同 MPD 框架与相同引导设置下，DDIM 与 DPM-Solver++ 在成功率/可行率与推理时间上有何差异？
- RQ2（速度–质量权衡）：随着采样步数变化，二者的轨迹质量与推理延迟的权衡曲线如何变化？是否存在稳定的“拐点（knee point）”可作为推荐运行点？
- RQ3（难度与分布偏移）：在 6 个难度梯度场景中，采样器偏好是否随场景难度改变？在新增障碍（OOD）设置下，差异是否会显著扩大或反转？
- （可选）RQ4（消融归因）：DPM-Solver++ 的关键设置（order、multistep/singlestep、skip_type 等）对结果的影响有多大？是否存在一组对多数场景都较稳健的默认配置？

## 六、研究假设（Hypotheses）
- H1（效率假设）：在训练环境中，DPM-Solver++ 在达到同等成功率（或同等轨迹质量）时更省时（端到端推理时间更短），或在相同时间预算下能达到更高成功率/更好质量。
- H2（拐点假设）：对每个场景，成功率/质量随采样步数增加会出现边际收益递减；且 DPM-Solver++ 的拐点往往出现在更少的采样步数或更小的时间预算附近。
- H3（OOD 假设）：在新增障碍（OOD）设置下，所有采样器性能都会下降；若不进行针对 OOD 的额外调参，DPM-Solver++ 相对 DDIM 的优势可能减弱，甚至在部分高难度场景出现劣化。

## 七、研究目标（Objectives）
- O1（实现与对齐）：在 MPD 代码中实现/对齐“可插拔 sampler”，确保 DPM-Solver++ 与 DDIM 走同一规划管线、同一计时边界，保证计时与评测口径一致。
- O2（六场景系统评测）：在 6 个场景上，对每个采样器 × 采样步数组合进行配对实验，统计成功率/可行率、轨迹质量与端到端推理时间。
- O3（曲线与推荐点）：绘制 success–time、quality–time 等曲线并识别拐点，形成“时间预算→推荐采样器与步数”的规则或查表。
- O4（消融与解释）：在少量代表性场景上做消融（order、multistep 等），给出对现象的定性解释与工程建议（本科论文以可解释的实验观察为主，不追求复杂理论证明）。

## 八、研究方法与实验设计（Methodology）

### 8.1 对照原则：只改采样器
实验中固定不变项包括：扩散模型结构与权重、MPD 规划流程（采样→引导/代价→碰撞检查/可行性筛选→输出轨迹）、机器人与仿真设置、场景几何与评测流程。唯一改变因素为采样器算法及其采样步数设置。这样可将性能差异主要归因于 sampler 本身。

### 8.2 论文口径与代码口径对齐（关键设定说明）
1）环境设置分两类：  
- Training environment：`env_id_replace: null`（无新增障碍）。  
- Additional objects（OOD）：将 `env_id_replace` 设为对应 `*ExtraObjects*` 环境（新增障碍，用于测试泛化/鲁棒性；不作为主调参目标）。

2）统计口径参考论文：以“contexts × trajectories”为单位统计，其中：  
- context：一个 start/goal 任务实例；  
- trajectories：在该 context 内采样/优化得到的候选轨迹集合。  
最终报告以跨 contexts 的 mean/std 为主（与论文更一致）。

3）公平性注意点：在 `*ExtraObjects*` 环境下，部分 start/goal 可能因碰撞而被代码跳过，导致实际评测到的 contexts 集合发生偏移。为提升可复现与公平性，最终对比优先使用固定的 contexts 列表文件（见 8.4）。

### 8.3 场景与配置（项目提供的 6 个场景）
本研究不新增场景，直接使用项目 `mpd-splines-public/scripts/inference/cfgs` 下的 6 个配置文件（按原项目设定的难度梯度）：
1. `config_EnvSimple2D-RobotPointMass2D_00.yaml`
2. `config_EnvNarrowPassageDense2D-RobotPointMass2D_00.yaml`
3. `config_EnvPlanar2Link-RobotPlanar2Link_00.yaml`
4. `config_EnvPlanar4Link-RobotPlanar4Link_00.yaml`
5. `config_EnvSpheres3D-RobotPanda_00.yaml`
6. `config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml`

每个场景分别在 `env_id_replace: null` 与 `env_id_replace: *ExtraObjects*` 两类环境设置下报告结果（对应论文 training vs additional objects 的呈现方式）。

### 8.4 start/goal 抽样、随机性与可复现性控制
对比实验中显式控制以下参数（保证公平与可复现性）：
- `selection_start_goal`：最终评测使用 `"validation"`（或直接指向一个固定的 contexts YAML 文件），避免使用 training 集导致乐观偏差。
- `seed`：同时影响“抽到哪些 contexts”与“扩散采样噪声”，因此同一轮对比中所有采样器应使用相同的 seed（或相同 seed 集合）。
- `n_start_goal_states`：每次评测包含的 contexts 数量。为降低方差，优先增大 `n_start_goal_states`，而不是用很多 seed 且每个 seed 只测极少 contexts。
- `n_trajectory_samples`：每个 context 内采样的轨迹条数（默认 100；调参阶段可降低以节省算力）。

固定 contexts 的建议做法：先生成一份 contexts YAML（例如 100 个 context），然后所有 sampler/超参配置都用同一份文件作为 `selection_start_goal`。若希望 training env 与 OOD 使用严格可行的 contexts，可分别为两类环境各生成一份 contexts 文件并分别报告（也更贴近论文口径）。

### 8.5 采样器与采样步数/时间设置
对比对象：
- Baseline：DDIM（MPD 默认，`diffusion_sampling_method: ddim`）
- 主要方法：DPM-Solver++（`diffusion_sampling_method: dpm_solver_pp`；重点关注 order=2、multistep）
- （可选）UniPC（`diffusion_sampling_method: unipc`）与/或 DDPM（`diffusion_sampling_method: ddpm`）作为参考点

由于不同采样器的“一步”计算含义并不完全一致，且端到端推理还包含引导与碰撞检查等开销，本论文以端到端推理时间作为主要对比口径；采样步数作为可控变量，用于画出速度–质量曲线。

步数设置：对每个采样器分别设置 6–8 个离散步数点覆盖低/中/高区间（例如 DDIM 的 `ddim_sampling_timesteps`、DPM-Solver++ 的 `dpm_solver_steps`、UniPC 的 `unipc_steps`），并在同一硬件与同一计时边界下记录端到端推理时间。

### 8.6 实验路线图（调参 → 基准对比 → 消融）
A）参数调优（快为主，不在 OOD 上“调到最好”）  
- 主环境：`env_id_replace: null`；用 `*ExtraObjects*` 做少量 sanity check（鲁棒性约束）。  
- 建议规模：`n_start_goal_states=10~20`，`seeds=1~3`（固定小集合），必要时将 `n_trajectory_samples` 暂降到 20/32 以避免高维场景耗时过大。

B）最终基准对比（对齐论文口径）  
- 覆盖 6 个场景；两类环境（training / OOD）都报告。  
- 建议规模：`n_start_goal_states=100`，`n_trajectory_samples=100`；seed 可固定 1 个或使用 3 个 seed 增强严谨性。  
- 优先使用固定 contexts 文件保证“同题同考”。

C）消融实验（解释“为什么这个 solver 更好/更稳”）  
- 选 1 个高难（如 Panda spheres/warehouse 的 OOD）+ 1 个低维（2D/planar）作为代表。  
- 对比维度：order=1 vs 2、multistep vs singlestep、dpmsolver vs dpmsolver++、不同 skip_type 等。  
- 建议规模：`n_start_goal_states=50~100`，`seeds=1~3`。

### 8.7 指标（Metrics）
至少包含三类指标（尽量沿用项目已有日志字段）：
- 成功率：满足“到达目标 + 无碰撞 + 约束满足”的 trial 占比；必要时可同时报告宽松成功（仅无碰撞到达）用于补充分析。
- 可行率/碰撞相关：如 `fraction_valid`、`collision_intensity` 等，用于刻画 OOD 场景下的退化程度。
- 轨迹质量：路径长度、平滑性（速度/加速度平方积分或项目已有 smoothness 指标）、最小障碍物间隙、末端误差等。
- 推理延迟：端到端 wall-clock time（可拆分 sampling / guidance / collision-check，若项目已支持）。

### 8.8 统计与分析方法
- 统计：跨 contexts 汇总 mean/std，并提供最差/最好 case（min/max）辅助分析鲁棒性。
- 曲线：对每个场景画 success–time、quality–time、success–steps、time–steps 曲线。
- 拐点：用边际收益下降（差分斜率阈值）定位 knee point，形成推荐步数与时间预算范围。
- 对比：重点观察“随场景难度变化的差异趋势”，并对 training vs OOD 分开讨论。

## 九、预期成果与呈现形式
1）六场景总表：每个（场景 × 步数设置）下 DDIM vs DPM-Solver++（±可选 UniPC/DDPM）的成功率/可行率/质量/时间对照。  
2）六组曲线：每个场景一组 time–success 与 time–quality，并标注 knee point。  
3）推荐规则：输入=场景类别（2D/planar/panda）与时间预算，输出=推荐采样器与步数设置。  
4）讨论与局限：对 OOD 场景退化原因进行定性分析，并说明本科工作范围内的限制与后续改进方向。

---

## 十、提炼版（便于写作与答辩放在文末）

### 10.1 研究问题（提炼）
- RQ1：在相近推理时间（或相同时间预算）下，DDIM vs DPM-Solver++ 谁更快/更成功？
- RQ2：两者的速度–质量曲线与拐点位置有何不同？
- RQ3：这种差异是否随场景难度、以及 training vs OOD（additional objects）而变化？

### 10.2 研究假设（提炼）
- H1：DPM-Solver++ 在相同时间预算下能达到与 DDIM 相近或更好的成功率/质量。
- H2：DPM-Solver++ 的拐点（边际收益开始明显下降的点）往往更靠近低步数/低耗时区域。
- H3：在 OOD（新增障碍）下优势可能减弱或反转。

### 10.3 实验方法（提炼）
- 自变量：采样器类型（DDIM、DPM-Solver++、可选 UniPC/DDPM）、采样步数档位、环境设置（training vs OOD）。
- 因变量：成功率/可行率、轨迹质量指标、端到端推理时间。
- 控制变量：模型权重、引导项、规划管线、场景配置与评测流程固定不变。
- 数据与场景：6 个配置文件（2D/planar/panda 各类任务），分别在 `env_id_replace: null` 与 `*ExtraObjects*` 下评测。
- 采样与随机性：最终评测使用固定 contexts 文件（或 `"validation"`），同一轮对比统一 seed；`n_start_goal_states` 优先设大（最终建议 100），每个 context 内 `n_trajectory_samples` 默认 100。
- 分析输出：每场景的曲线与拐点、跨场景汇总表、按时间预算给出推荐采样器与步数的规则。

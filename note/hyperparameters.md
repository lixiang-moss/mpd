
这是根据你提供的信息整理好的 Markdown 格式文档。它详细解释了 DPM-Solver 配置参数及其对采样轨迹、速度和约束引导（Guidance）的影响。

---

# DPM-Solver 与 Cost Guidance 配置详解

此配置控制 **DPM-Solver(++)** 的扩散采样过程，以及在采样末段进行的**基于代价 (Cost) 的梯度引导**。该机制通过梯度优化，将生成的轨迹控制点推向“更低代价 / 更少碰撞 / 更满足约束”的方向。

**相关代码实现：**

* `dpm_solver_sampler.py` (Line 33)
* `sample_functions.py` (Line 93)

---

## 1. 引导介入时机 (Guidance Timing)

控制什么时候开始对采样轨迹施加约束引导。

### `t_start_guide_steps_fraction`

定义引导从采样的“后半段/末段”开始的比例。

* **计算逻辑**：实际使用的引导步数为 `t_start_guide = ceil(fraction * dpm_solver_steps)`（见 `inference.py` Line 330）。
* **生效范围**：在 DPM-Solver 的最后 `t_start_guide` 个 solver step 会执行 guidance。即满足 `(dpm_solver_steps - step) <= t_start_guide` 时生效。
* **调节影响**：
* **调大**：Guidance 更早开始，覆盖步数更多。
* *优点*：约束更容易被满足。
* *缺点*：速度变慢；更可能“偏离生成先验”，导致轨迹质量下降或不稳定。


* **调小**：Guidance 仅在接近收敛的末段进行。
* *优点*：生成速度更快，轨迹更稳定。
* *缺点*：可能来不及把发生碰撞或违反约束的轨迹“推”回去。





---

## 2. DPM-Solver 采样器核心参数

控制采样器本身的速度、质量和数值稳定性。

### `dpm_solver_steps`

DPM-Solver 的总函数评估次数 (NFE)，即总“步数”。

* **增大**：更慢，但通常更接近真实反向 ODE 解，样本质量更好、更稳。
* *注意*：末段 guidance 的总次数也会随之增加（受 `t_start_guide_steps_fraction` 影响）。


* **减小**：更快，但误差变大。Guidance 往往更难在有限步数内“救回来”。

### `dpm_solver_order`

求解器的阶数（1, 2, or 3）。

* **特点**：高阶通常单位步长更准，但在“强引导 / 强修正”场景下更容易不稳定。
* **建议**：官方经验推荐在强引导场景下使用 **`order=2`** 配合 `multistep`。

### `dpm_solver_method`

步进策略（来自官方实现 `dpm_solver_pytorch.py` Line 1047）。

* **`'multistep'`**：多步法，利用历史点做高阶更新。**强引导常用**，一般更稳。
* **`'singlestep'`**：单步法的组合（论文中的 fast 模式）。适用于无引导或弱引导。
* **`'singlestep_fixed'`**：固定阶单步法。较“机械”，若 `steps` 不是 `order` 的整数倍可能浪费 NFE。
* *(注：官方的 `'adaptive'` 模式因启用了 `correcting_xt_fn` 做硬约束在此代码中被禁止)*

### `dpm_solver_algorithm_type`

算法核心类型。

* **`'dpmsolver++'`**（当前选择）：内部使用 data prediction /  形式。通常质量更好，在强引导下更稳。
* **`'dpmsolver'`**：内部使用 noise prediction /  形式。

### `dpm_solver_skip_type`

时间步的分布方式，决定“每一步对应的噪声水平”。

* **`'time_uniform'`**： 均匀分布。
* **`'logSNR'`**：logSNR 均匀分布。
* **`'time_quadratic'`**： 的二次分布。通常会偏向在末段（低噪声区）进行更细致的采样。
* **影响**：改变此参数意味着同样的“最后 N 步 guidance”实际发生在不同的噪声区间，会显著影响引导的效果和稳定性。

### 其他微调参数

* **`dpm_solver_solver_type`**：`'dpmsolver'` (推荐) vs `'taylor'`。
* **`dpm_solver_denoise_to_zero`**：是否最后额外做一次“到  的去噪”。
* *True*：末端更“干净”，但多一次模型调用（慢）。
* *False*：省一次调用。


* **`use_continuous_time`**：模型时间输入方式。
* *False*：将连续  映射并 round 到离散步（更贴近训练分布）。
* *True*：将  作为连续值输入（贴近 ODE 假设，但在离散训练的模型上可能有分布外风险）。


* **`lower_order_final`**：末几步是否降阶以求稳定。在步数较多（如 76 步）时通常无影响。

---

## 3. 代价引导 (Guidance) 相关参数

平衡“更像先验 (Prior)”与“更满足约束 (Constraints)”之间的博弈。
这些参数控制 `guide_gradient_steps()`：在每个被引导的 solver step 里，对当前  做 `n_guide_steps` 次梯度上升（实际上是 Cost 的梯度下降）。

### `prior_weight_with_guide`

当提供了 Guide 时，对扩散模型预测的  施加的整体缩放系数（`dpm_solver_sampler.py` Line 123）。

* **注意**：只要 `guide != None`，此缩放作用于**所有** solver steps，不仅限于末段。
* **`1`**：更信先验 / 更强去噪漂移。Guidance 相对更难把轨迹推开障碍。
* **`< 1`**：先验变弱，Guidance 更容易主导。
* *风险*：更可能破坏生成分布，导致出现怪异轨迹。



### `n_guide_steps`

每个被引导的 solver step 内，执行梯度更新的次数。

* **增大**：约束更容易满足，但更慢，且更可能过度修正。

### `guide_lr`

Guidance 的 SGD 学习率。

* **增大**：每步推得更猛。容易震荡、不稳，经常会直接撞上 `max_perturb_x` 上限。
* **减小**：更稳，但可能推不动，无法避障。

### `max_perturb_x`

每个 solver step 内，Guidance 允许  相对“Guidance 前的初始 ”的最大逐元素偏移量（`sample_functions.py` Line 154）。

* **增大**：Guidance 权限更大，更能避障/修约束。
* *风险*：更可能偏离先验。


* **减小**：更稳、更像先验。
* *风险*：可能来不及避障。



### 梯度裁剪 (Gradient Clipping)

防止梯度爆炸导致轨迹飞出流形。

* **`clip_grad`**：是否启用裁剪。*True* 通常更稳（碰撞代价在接触/穿透附近梯度可能极大）。
* **`clip_grad_rule`**：裁剪策略。
* `'norm'`：按向量范数裁剪到 `max_grad_norm`（更保持方向）。
* `'value'`：逐元素裁剪到 `[-max, +max]`（更硬，更可能改变梯度方向）。


* **`max_grad_norm`**：配合 `'norm'` 使用，限制梯度向量大小。
* **`max_grad_value`**：配合 `'value'` 使用，限制梯度分量绝对值。

### `scale_grad_by_std`

是否用扩散过程的方差 (`model_var`) 对 Guidance 梯度做缩放（`sample_functions.py` Line 142）。

* **True**：Guidance 强度随噪声水平变化。
* *问题*：通常需要重新调 `guide_lr`。如果你只在末段（低噪声、方差小）才做引导，这可能会让 Guidance 变得非常弱。


* **False**：梯度直接使用。更直观，更好调参。



你想要什么？,这种参数怎么调？,代价是什么？
轨迹更顺滑、更像人走的,调大 prior_weight，调小 guide_lr，调小 max_perturb_x,可能会撞到障碍物（约束满足率低）。
死命令，必须避开障碍,调小 prior_weight，调大 guide_lr，调大 steps，提前介入（调大 fraction）,轨迹可能扭曲、怪异，计算变慢。
计算速度要快,调小 steps，减少 n_guide_steps,精度下降，可能来不及避障。
避障经常失败（撞墙）,检查 guide_lr 是否太小（推不动）或太大（推飞了），或者 max_perturb_x 卡太死。,需要反复微调力度。
轨迹画出了锯齿/乱跳,guide_lr 太大了，或者 Solver order 太高（惯性太大）。,减小力度，或者改用更稳的 Solver 配置。
Evaluation of diffusion samplers for
manipulator motion planning
Xiang Li
1 abstract
Diffusion models have shown promise in manipulator trajectory planning, but their
high inference cost limits practical use. The choice of sampling strategy strongly
affects both efficiency and solution quality, yet systematic comparisons in manipulator
tasks are still limited. This thesis proposes a general evaluation framework to
compare different diffusion samplers under unified conditions. The framework
is based on a modular design with Hydra-based configuration management and
is evaluated using Motion Planning Diffusion (MPD), where the performance of
different sampling strategies is quantitatively analyzed and compared.
2 Introduction
Diffusion-based methods for manipulator motion planning have attracted increasing
attention in recent years[1]. By generating trajectories through iterative denoising,
diffusion models can represent multimodal distributions and address complex high-
dimensional planning problems[2], [3]. Representative approaches such as MPD
leverage diffusion models to generate collision-free trajectories in novel environ-
ments[4]. However, a well-known limitation of diffusion models lies in their slow
sampling process[2].
To meet real-time planning requirements, various accelerated sampling strategies
have been proposed. For example, standard Denoising Diffusion Probabilistic Models
(DDPM)[2] sampling requires hundreds of denoising steps, Denoising Diffusion
Implicit Models (DDIM)[5] introduces deterministic sampling to reduce inference
steps, and higher-order solvers such as DPM-Solver++ [6] DDIM further approximate
the diffusion process with fewer iterations. These samplers exhibit different trade-
offs between sampling speed and trajectory quality: faster sampling often comes
at the cost of degraded solution quality, while higher-quality trajectories usually
require longer inference time.
As pointed out in recent surveys, “finding an appropriate trade-off between sampling
quality and inference speed under task constraints remains an open problem, and only
a limited number of studies have systematically compared DDPM, DDIM, or other
samplers in robotic manipulation tasks[1] .” This indicates a lack of unified evaluation
protocols for comparing diffusion samplers in manipulator motion planning. The
2 Introduction
absence of such benchmarks makes it difficult to quantitatively assess the advantages
and limitations of different sampling strategies, and complicates the selection of
appropriate samplers and hyperparameters for practical applications.
Motivated by this gap, this work aims to construct a general evaluation framework
to systematically compare diffusion-based samplers in manipulator motion planning.
Unlike prior studies that focus on a single algorithm or specific task, the proposed
framework supports multiple samplers, hyperparameter configurations, and task
environments in a unified evaluation pipeline. An automatic hyperparameter tuning
mechanism is incorporated to ensure fair comparison among samplers. Moreover,
Hydra-based configuration management is employed to decouple experimental
settings from implementation details, enabling non-intrusive integration with MPD
or other diffusion-based planners through standardized interfaces. This design
significantly improves the reusability and extensibility of the evaluation framework.
This paper first formulates the research questions and objectives, then describes the
proposed methodology including experimental design, hyperparameter optimization,
and statistical analysis, and finally outlines the structure of the thesis.
3 Research Questions and Hypotheses
RQ1: How can we design a decoupled, extensible, and reproducible evaluation
framework to fairly compare different diffusion samplers and their parameter con-
figurations in manipulator tasks?
• H1.1 (Reproducible configuration and execution): Using Hydra for hierar-
chical configuration management together with a subprocess-based CLI execu-
tion workflow improves reproducibility by enabling fully specified experiment
configurations (including sampler choices, parameters, seeds, and scenarios)
and by producing consistent, repeatable runs from the same configuration.
• H1.2 (Separation of experiment execution and metric computation): De-
coupling experiment execution and output parsing from metric computa-
tion—where metrics are computed within the planner/model under a shared,
predefined metric specification—reduces discrepancies introduced by incon-
sistent evaluation logic and improves cross-sampler comparability under a
unified protocol.
RQ2: How can the evaluation framework support an efficient, automated, and
reproducible hyperparameter tuning process such that each sampler is evaluated
under a representative configuration for fair comparison?
• H2.1 (Tuning beats defaults): Under a unified evaluation protocol, automati-
cally tuned configurations consistently outperform default or ad-hoc settings.
Research Questions and Hypotheses 3
• H2.2 (Efficiency and consistency): Within a predefined task-specific search
space, automated tuning enables efficient and well-documented parameter
selection with minimal manual intervention, and produces more consistent
results across random seeds or repeated runs, improving reproducibility and
interpretability.
RQ3: How can we construct a multi-metric evaluation and ranking scheme that is
fair, interpretable, and adaptable to different task preferences?
• H3.1 (Category-based reporting and normalization): Grouping metrics into
categories and normalizing them with respect to a baseline sampler improves
comparability across scenarios and supports rankings that better reflect speed–
quality trade-offs.
• H3.2 (Preference-aware scoring): Providing configurable scoring functions
and weights enables preference-aware comparisons (e.g., prioritizing success
vs. speed) and yields stable recommendations for top-performing samplers
across a reasonable range of preference settings.
4 Research Objectives
To address the above research questions, this work defines the following objectives:
• O1: Evaluation Framework Design and Implementation. Design and im-
plement a modular and automated evaluation framework based on Hydra
configuration management. The framework interfaces with the evaluated mod-
els through standardized interfaces, enabling seamless evaluation of different
samplers and task settings. It includes modules for configuration loading,
sampler tuning, experiment execution, result parsing, and data aggregation,
supporting a reproducible “single-call, multi-run” workflow.
• O2: Automated Hyperparameter Optimization. Integrate automatic hyper-
parameter tuning mechanisms (e.g., hybrid Grid Search and Bayesian Opti-
mization) to identify near-optimal parameters for each sampler. This objective
aims to verify the “fairness” of comparisons by ensuring that each sampler is
evaluated under its best-performing configuration.
• O3: Multi-dimensional Metrics and Ranking Scheme. Establish a compre-
hensive metric system and a flexible ranking scheme. This involves:
– Defining standardized metrics covering success rate, trajectory quality,
and solution diversity.
– Developing a composite scoring mechanism based on normalization
and weighted aggregation, allowing for interpretable comparisons and
4 Research Objectives
rankings adaptable to different user preferences (e.g., latency-critical vs.
safety-critical).
• O4: Experimental Validation and Selection Guidelines. Apply the proposed
framework to multiple manipulation scenarios to evaluate mainstream diffu-
sion samplers. The objective is to quantitatively analyze trade-offs, validate
the reproducibility of the framework, and derive empirical guidelines to
assist practitioners in selecting the most appropriate sampling strategies for
specific manipulator tasks.
5 Methodology
This study constructs a reproducible evaluation framework for hyperparameter
optimization of individual samplers and systematic comparison of different samplers
under optimized settings. All experiments are managed via Hydra configuration
files. The execution module launches the MPD inference process as a subprocess
according to the configuration. After execution, the framework parses logs and
outputs, stores structured metrics, performs statistical analysis and visualization,
5.1 Evaluation Framework Overview
The proposed evaluation framework consists of the following core components:
• Configuration management module : manages experimental parameters and
task settings;
• Experiment execution module: controls execution and scheduling of inference
runs;
• Automatic hyperparameter tuning module: searches for near-optimal sampler
parameters;
• Result parsing module: extracts structured metrics from model outputs;
• Statistical analysis and visualization module: generates comparative results
and figures.
The overall workflow follows a two-stage evaluation paradigm:
• Stage 1: Sampler Hyperparameter Tuning Identify near-optimal parameter
configurations for each sampler.
• Stage 2: Fair Sampler Comparison Evaluate different samplers under their
respective optimized settings across multiple scenarios.
This design ensures that all samplers are compared under their best-performing
configurations, thereby improving the fairness and credibility of the evaluation
results.
Methodology 5
5.2 Stage I: Sampler Hyperparameter Optimization
5.2.1 Search Space and Optimization Strategy
For each sampler, a dedicated hyperparameter search space is defined, including
both discrete options (e.g., solver variants, solver order) and continuous parameters
(e.g., guidance steps,). Different optimization strategies are adopted depending
on the characteristics of the search space. Grid search or random search is applied
for small discrete spaces, while Bayesian optimization is employed for scenarios
involving continuous variables or limited computational budgets.
5.2.2 Objective Function and Constraints
The hyperparameter optimization follows a threshold-and-optimization paradigm.
Specifically, a feasibility threshold is first enforced, such as requiring the success
rate or validity ratio to exceed a predefined value. Among the configurations that
satisfy this constraint, further optimization is performed with respect to efficiency
and trajectory quality metrics.
5.2.3 Output of the Optimization Stage
After the completion of Stage I, the framework outputs the optimal (or top-k) hy-
perparameter configurations for each sampler. These configurations are stored as
tuned configuration files in Hydra YAML format. In addition, structured result ta-
bles are generated and saved, providing detailed records of the tuning process and
performance trends.
5.3 Stage II: Comparative Evaluation with Tuned Samplers
In Stage II, each sampler is evaluated using the optimal hyperparameters obtained
from Stage I. The tuned configurations are directly loaded, and the samplers are
executed under a unified evaluation protocol across a broader set of scenarios. The
execution, result parsing, storage, and statistical analysis procedures remain identical
to those used in Stage I.
5.4 Evaluation Metrics and Statistical Analysis
To systematically evaluate the performance of different diffusion sampling strategies
for manipulator motion planning, this study constructs an evaluation metric system
from five dimensions: feasibility, accuracy, trajectory quality, diversity, and computa-
tional efficiency. All metrics are computed under a unified experimental setup to
ensure fair comparisons across samplers.
(1) Feasibility and Safety.
These metrics are used to determine whether planning succeeds and to analyze the
primary sources of failure; they serve as the basic filtering conditions for subsequent
statistical analyses.
6 Methodology
• Success rate: whether a feasible, executable trajectory is produced; aggre-
gated as the overall success rate.
• Validity ratio: the fraction of generated trajectories that pass constraint checks
(e.g., collision-free, constraint-satisfying); measures robustness in producing
usable solutions.
• Collision intensity: a continuous measure of collision risk/severity (lower is
better).
(2) Goal Reaching Accuracy.
This part is computed only on the set of successful or valid trajectories, and is used
to measure the terminal-state accuracy of the planning result.
• End-effector position error: the final end-effector position error (norm) of
the trajectory.
• End-effector orientation error: the final end-effector orientation error (norm)
of the trajectory.
(3) Trajectory Quality.
Used to measure the kinematic reasonableness and executability of generated trajec-
tories, computed only on successful or valid samples.
• Path length: path length of the trajectory.
• Smoothness: smoothness of the trajectory.
(4) Coverage and Diversity.
These metrics are used to analyze a sampler’s exploration capability in the solution
space:
• Trajectory diversity: diversity of the valid trajectory set, reflecting coverage
and potential mode-collapse risk.
(5) Efficiency and bottleneck decomposition. This part analyzes the practical usability
of sampling strategies and is the core basis for the speed–quality trade-off.
• Total inference time: end-to-end inference time.
• Sampling time: time spent in the diffusion sampler / generator.
• Guidance time: time spent in cost guidance
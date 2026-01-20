% main.tex
\documentclass[]{ccs-proposal}
\providecommand{\TODO}[1]{}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\setlist[enumerate]{itemsep=0pt, topsep=0pt, parsep=0pt, partopsep=0pt}
\usepackage{comment}
\usepackage{parskip}
%\usepackage[backend=biber,style=numeric,sorting=none,doi=false,isbn=false,url=false]{biblatex}

\title{Evaluation of diffusion samplers for manipulator motion planning}
\begin{comment}
\group{Open Distributed Systems (ODS)}
\author{Xiang Li}
\birthday{11. November 1980}
\birthplace{Berlin}

\thesistype{Bachelor Thesis in Computer Science}\thesiscite{Bachelor Thesis~(Bachelorarbeit)}
\advisors{Jiangtao Shuai}
\referees{Prof. Dr. Manfred Hauswirth}
\end{comment}

\begin{document}
\maketitle



\section{abstract}


Diffusion models show promise for robot motion planning, but their iterative denoising causes significant latency. Although accelerated samplers such as DDIM can reduce steps, their advantages in robotic planning remain largely untested in systematic comparisons.
This thesis proposes a systematic evaluation of different diffusion samplers within a fixed Motion Planning Diffusion (MPD) framework. 
By varying the number of sampling steps and measuring planning success rate, trajectory quality, and inference time, the study will quantify the speed--quality trade-off. The outcome will be practical guidelines for selecting a sampler and step budget to meet real-time constraints while maintaining plan quality.





\section{Introduction}
Generative diffusion models (DMs), e.g., DDPM \cite{ho2020ddpm}, have shown promise for motion generation in robotics due to their ability to represent multi-modal trajectory distributions. Notably, \emph{Motion Planning Diffusion (MPD)} demonstrates that a learned diffusion prior over trajectories can produce feasible collision-free motions in novel scenes \cite{carvalho2023mpd}. Despite these advances, diffusion planning typically requires tens to hundreds of iterative denoising steps, leading to slow inference compared to classical planners (e.g., RRT*, CHOMP)~\cite{karaman2011sampling,ratliff2009chomp}. In time-sensitive setups, balancing trajectory quality and inference time becomes critical.
Accelerated samplers aim to reduce steps and latency. DDIM \cite{song2021ddim} introduces deterministic implicit sampling that often achieves good quality with fewer steps. Higher-order ODE-based methods such as DPM-Solver/++ \cite{lu2022dpmsolver,lu2022dpmsolverpp} exploit the probability flow ODE to take larger steps with controlled error. While these samplers are effective in image generation, robotic planning studies,especially those focusing on manipulators,often lack systematic comparisons across samplers and step budgets \cite{wolf2025survey}. This proposal targets that gap.

\textbf{Problem Statement.}
Given a fixed diffusion motion planner and task setting, how do different samplers and step counts impact success rate, path quality, and computation time? Answering this enables principled selection of sampler/steps for time-constrained robotic deployment.

\section{Research Questions }

  \textbf{RQ1. }Under a selected diffusion-based planner (MPD) and identical guidance, how do DDPM, deterministic DDIM, and DPM-Solver++ (a higher-order ODE solver) compare in terms of trajectory success rate and inference latency when operating under similar numbers of function evaluations (NFE)?\\
  \textbf{RQ2. }How do the number of sampling steps (or equivalently, NFE) shape the relationship between trajectory quality and planning speed, and how can an optimal trade-off curve be defined?\\
  \textbf{RQ3. }Along a task-difficulty gradient---from simple to complex scenarios (e.g., increasing obstacle number/density, narrower passages, proximity to reachability boundaries)---do the preferred samplers change, and is there a significant \emph{sampler~$\times$~difficulty} interaction effect?



\section{Objectives}
The overall objective of this thesis is to provide quantitative guidelines for balancing speed and quality in diffusion-based motion planning for manipulator. To this end, the thesis pursues the following specific objectives:\\

  %\textbf{O1. Unified diffusion-planning framework.} Implement an MPD-based system in simulation with a modular ``sampler plugin'' interface that allows DDPM, DDIM, and DPM-Solver++ to be interchanged under a fixed diffusion model and environment.\\
  \textbf{O2. Evaluation across a ``step budget~$\times$~difficulty gradient'' design.} For each sampler and step budget, conduct systematic experiments on multi-level difficulty scenarios (e.g., S1: simple---wide passages / few obstacles; S2: medium---narrower passages / more obstacles; S3: difficult---tight passages / near reachability limits), and measure planning success rate, trajectory quality (e.g., smoothness, path length, minimum clearance, constraint violations), and inference time.\\
  \textbf{O3. Characterization of the speed--quality trade-off.} For each sampler and difficulty level, construct time--quality curves, identify characteristic ``knee points'' of diminishing returns, and analyze how these frontiers and knee points shift as task difficulty increases.\\
  \textbf{O4. Practical recommendations.} Based on the empirical results, derive sampler and step-budget selection guidelines for different combinations of real-time constraints and task difficulty, and summarize them in a concise decision procedure for practitioners.\\
  %\textbf{O5. Open-source evaluation toolkit.} Package and release the evaluation code, configurations, and logs as an open-source suite, enabling reproducibility and straightforward extension to additional samplers and planners.



\section{Expected Contributions}

  \textbf{Fair, difficulty-aware sampler benchmark.} \\A reproducible benchmark that evaluates DDPM, DDIM, and the DPM-Solver family under matched function evaluation counts (NFE), a unified noise schedule, and fixed guidance. The benchmark uses paired trials and stratified scenario difficulty levels (simple / medium / hard), with environment and timing controlled to enable like-for-like comparisons.\\
  \textbf{Speed--quality trade-off characterization with knee points.}\\ Empirical curves that relate success rate and trajectory quality to wall-clock inference time across combinations of \{sampler~$\times$~NFE~$\times$~difficulty\}, together with identification of diminishing-return knee points at each difficulty level. In addition, ``time-to-success'' under given time budgets is reported to support time-constrained decision-making.\\
  \textbf{Difficulty-aware practical guidance.} \\A sampler-selection strategy that offers both a ``conservative'' profile (favoring higher success rates) and a ``fast-usable'' profile (favoring lower latency), thereby supporting different real-time deployment requirements.\\
  \textbf{Open evaluation framework and resources.}\\ An open-source ROS~2/Gazebo evaluation suite that includes the MPD planner and a pluggable sampler interface, along with minimal configurations and example setups, so that experiments can be reproduced and the framework can be reused or extended with minimal effort.






\section{Methodology}
\subsection{System, Planner, and Guidance}

%All experiments will be conducted in a ROS~2 + Gazebo simulation environment with a 6-DoF LeRobot SO-101 manipulator model. All experiments use a fixed, pre-trained diffusion model over joint-space trajectories within the (MPD) framework. 

Across all experiments, the diffusion model (architecture and weights), the robot and environment setup, and the high-level MPD planning pipeline (trajectory sampling, collision checking, and execution) remain unchanged; only the sampling algorithm and its step budget (number of function evaluations, NFE) are varied. Collision avoidance, goal satisfaction, and other task-specific constraints are handled by a fixed set of cost and constraint terms that are kept identical for all samplers and step budgets, so that performance differences can be attributed to the choice of sampler and step budget.

\subsection{Samplers and Step Budgets}
We compare three samplers: DDPM , DDIM , and DPM-Solver++. All  samplers are implemented behind a common stepper API with a consistent counting of network forward passes. Whenever a sampler requires multiple model evaluations within a single integration step (for example, in higher-order methods), each forward pass is counted towards the total number of function evaluations (NFE). For each sampler, we predefine a small set of step budgets (e.g., very low, medium, and relatively high NFE) that span from ``very fast but potentially lower quality'' to ``slower but closer to reference performance.'' The exact NFE values will be chosen based on preliminary experiments and will lie in a meaningful range of tens to hundreds of effective steps.

\subsection{Scenario Design and Difficulty Gradient}
To address RQ3 and realize Objectives O2--O3, we construct a three-level task-difficulty gradient:
\begin{itemize}
\item \textbf{S1 -- Simple}: wide passages, few obstacles, and start--goal configurations well inside the reachable workspace;
\item \textbf{S2 -- Medium}: narrower passages, more obstacles, and start--goal pairs closer to cluttered regions;
\item \textbf{S3 -- Difficult}: very tight passages, higher obstacle density, and configurations near reachability boundaries or joint limits.
\end{itemize}
For each difficulty level, we define one canonical scene template (i.e., a fixed obstacle layout representative of that difficulty regime). Within each template, we sample multiple start--goal pairs inside task-specific bounds, and for a given start--goal pair the geometric setup is kept identical across samplers. This ``stratified by difficulty'' design supports a difficulty-aware benchmark and enables analysis of sampler $\times$ difficulty interaction effects.


\subsection{Experimental Protocol}
The experimental protocol is designed to support fair, paired comparisons between samplers and step budgets:
\begin{enumerate}
\item \textbf{Scene and start--goal selection:} For each difficulty level S1--S3, we use its canonical scene template and generate multiple scene instances by sampling start--goal pairs.
\item \textbf{Paired trials:} For every pair (scene instance, start--goal configuration), we run all samplers (DDPM, DDIM, DPM-Solver++) under all selected step budgets. This produces paired results in which every condition is evaluated under exactly the same geometric conditions, reducing variance and enabling one-to-one comparisons.
\item \textbf{Fair timing:} For each trial, we record wall-clock planning time from the start of diffusion sampling to the final collision-checked trajectory. Timing includes all model evaluations and collision checks, ensuring that ``faster'' results do not arise from omitting necessary computations.
\item \textbf{Success criteria:} A trial is considered successful if it reaches the goal within the allowed planning horizon, remains collision-free, and satisfies task constraints (e.g., joint limits). If a trial fails, its time is still recorded, and the failure is reflected in the success-rate metrics.
\item \textbf{Repetitions:} To obtain stable statistics, each combination of sampler, NFE, and difficulty level is evaluated over multiple scene instances and multiple independent trials. These repetitions are later aggregated into means and uncertainty ranges.
\end{enumerate}


\subsection{Metrics and Statistical Analysis}
For each sampler, step budget (NFE), and difficulty level, we compute the following metrics:
\begin{itemize}
\item \textbf{Success rate}: the proportion of trials that produce valid, collision-free trajectories reaching the goal;
\item \textbf{Trajectory quality}: including trajectory length, simple smoothness measures (e.g., integrated squared velocity or acceleration), minimum distance to obstacles, and any constraint violations;
\item \textbf{Inference time}: the measured planning time as defined above.
\end{itemize}
For each experimental condition, we report mean values and simple uncertainty ranges (e.g., confidence intervals or standard errors) to reflect variability. To answer RQ1--RQ3 and achieve Objectives O2--O4, we perform the following analyses:
\begin{itemize}
\item \textbf{Within-sampler trade-offs}: For each sampler and difficulty level, we plot success rate versus NFE and time versus NFE to observe how performance evolves as the step budget increases.
\item \textbf{Time--quality curves}: For each sampler and difficulty level, we construct time--quality curves that relate wall-clock time to success rate and trajectory-quality metrics, revealing speed--quality frontiers where additional computation yields diminishing returns.
\item \textbf{Knee-point identification}: Using simple slope-based heuristics (e.g., detecting where marginal gains in quality per unit time drop sharply), we identify characteristic ``knee points'' on the time--quality curves and treat them as candidate operating points with favorable speed--quality trade-offs.
\item \textbf{Time-to-success under budgets}: For a few illustrative time budgets, we determine, for each sampler and difficulty level, the smallest NFE that achieves a target success-rate threshold. This yields time-to-success profiles that directly support deployment decisions under real-time constraints.
\end{itemize}





\section{Outline}

% Requires \usepackage{enumitem} in the preamble for custom labels.
\begin{enumerate}[label=\arabic*.]
  \item Introduction
    \begin{enumerate}[label=\arabic{enumi}.\arabic*.]
      \item Motivation
      \item Problem statement and gap in sampler / step-budget comparisons
      \item Research questions and objectives
      \item Structure of the thesis
    \end{enumerate}

  \item Related Work
    \begin{enumerate}[label=\arabic{enumi}.\arabic*.]
      \item Diffusion-based motion planning
      \item Diffusion for decision-making and policies
      \item Accelerated sampling and distillation in diffusion models
      \item Benchmarks and evaluation practices 
      %for diffusion models in robotic manipulation
    \end{enumerate}

  \item Foundations
    \begin{enumerate}[label=\arabic{enumi}.\arabic*.]
      \item Diffusion models for generative modeling
      \item Motion planning and trajectory quality
      \item Collision-aware guidance in diffusion planning
    \end{enumerate}

  \item Approaches
    \begin{enumerate}[label=\arabic{enumi}.\arabic*.]
      \item System setup
      \item Samplers and step budgets
      \item Scenario design and difficulty gradient
      \item Experimental protocol
      \item Metrics and analysis plan
    \end{enumerate}

  \item Experiments
    \begin{enumerate}[label=\arabic{enumi}.\arabic*.]
      \item Experimental configuration
      \item RQ1: Sampler comparison at matched NFE
      \item RQ2: Effect of step budget on speed--quality trade-offs
      \item RQ3: Effect of difficulty gradient 
      \item Summary of empirical findings
    \end{enumerate}

  \item Discussion
    \begin{enumerate}[label=\arabic{enumi}.\arabic*.]
      \item Interpretation of sampler behavior and trade-offs
      \item Conservative vs.\ fast-usable operating profiles
      \item Comparison with classical planners and prior diffusion studies
      \item Limitations 
    \end{enumerate}

  \item Conclusion 
    \begin{enumerate}[label=\arabic{enumi}.\arabic*.]
      \item Summary of contributions and main findings
      \item Answers to the research questions
      \item Future work
    \end{enumerate}
\end{enumerate}




%implement Stepper API line124
\section{Work Plan (12 Weeks)}
\textbf{W1--W2:} Environment setup (ROS~2, Gazebo).\\
\textbf{W3--W4:} Integrate sampler modules, unify key settings, and perform basic correctness checks.\\
\textbf{W5--W6:} Instrument timing and logging; run small pilot experiments.\\
\textbf{W7--W9:} Full experiments; consolidate logs and plots.\\
\textbf{W10:} Statistical analysis; synthesize recommendations.\\
\textbf{W11--W12:} Writing and figures; proofreading; release code and reproducibility package.


%\newpage
\printbibliography
\end{document}

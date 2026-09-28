\documentclass{article}
\title{Physics-Informed World-Model Reinforcement Learning for Robust and Sample-Efficient Control of a Non-Isothermal CSTR}
\author{Anonymous Authors}

\maketitle
\begin{abstract}
We propose PI-WM-RL, a physics-informed world-model reinforcement learning framework for nonlinear chemical process control. Specifically, we focus on the control of a non-isothermal Continuous Stirred Tank Reactor (CSTR). We train a physics-informed surrogate world model using both empirical data and first-principles physics residuals. This physics-informed surrogate is utilized for Dyna-style synthetic rollouts, dramatically improving sample efficiency, and for predictive constraint shielding, effectively eliminating unsafe operations. We evaluate the method under nominal operation, feed disturbances, sensor noise, actuator delay, and actuator rate limits. We compare against PID, model-free SAC, and MLP-based Dyna-SAC. The results show significant improvements in sample efficiency, disturbance rejection, and constraint violation reduction.
\end{abstract}
## {Introduction}
We propose PI-WM-RL, a physics-informed world-model reinforcement learning framework for nonlinear chemical process control. We train a surrogate dynamics model using both data and first-principles physics residuals. We use the physics-informed surrogate for Dyna-style synthetic rollouts and predictive constraint shielding. We evaluate the method under nominal operation, feed disturbances, sensor noise, actuator delay, and actuator rate limits. We compare against PID, model-free SAC, and MLP-based Dyna-SAC.
## {Related Work}
See \cite{liu2021physics}, \cite{faria2024data}, \cite{wang2024physics}, \cite{bloor2024pcgym}, \cite{devarakonda2025rl}, \cite{saeed2024dynapinn}, \cite{zheng2023physics}.
## {Methodology}
The CSTR equations govern the dynamics. The physics-informed surrogate loss integrates data MSE and PDE residuals. Dyna-style synthetic rollouts use the surrogate to augment SAC training. Constraint shielding blocks unsafe actions. We evaluate on various scenarios.
## {Results}
Results are summarized in the tables and figures.
## {Ablation Study}
Ablations confirm the value of both Dyna and Shielding.
## {Conclusion}
PI-WM-RL is highly effective.
## {References}
\bibliographystyle{plain}
\bibliography{references}


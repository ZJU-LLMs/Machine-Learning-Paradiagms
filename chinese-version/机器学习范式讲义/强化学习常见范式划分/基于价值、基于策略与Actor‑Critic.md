# 基于价值、基于策略与Actor‑Critic

## 基于价值、基于策略与 Actor‑Critic 强化学习
### 背景
强化学习的核心是在与环境交互中学习最大化累积回报的策略。如何表示和优化策略，催生了三种基本范式：基于价值的方法（Value‑Based）、基于策略的方法（Policy‑Based）以及结合二者的 Actor‑Critic 方法。

基于价值的方法最早成熟，源于时序差分学习和动态规划思想。它隐式地定义策略——通过估计状态或状态‑动作对的价值，然后从中推导出最优动作（如贪心选择最大 Q 值的动作）。这类方法在离散动作空间中极为成功，但难以处理连续动作或随机策略。

基于策略的方法则直接参数化策略，通过优化期望回报来调整策略参数。它天然支持连续动作空间和显式探索，但梯度估计的方差较大，导致收敛缓慢。为降低方差，通常引入价值函数作为基线，这便自然过渡到 Actor‑Critic 方法。

Actor‑Critic 方法维护两个结构：Actor（策略网络）决定动作，Critic（价值网络）评估动作的好坏。Critic 为 Actor 提供低方差的梯度信号，而 Actor 则根据 Critic 的评估不断改进策略。这一框架兼具基于价值方法的样本效率和基于策略方法的灵活性，已成为现代深度强化学习的主流范式。Sutton 和 Barto (2018) 对这三种范式进行了系统阐述 [1]。

### 问题定义
强化学习问题通常被形式化为马尔可夫决策过程 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$，目标是找到策略 $\pi$ 最大化期望累积折扣回报 $J(\pi) = \mathbb{E}_{\pi}[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)]$。

+ **基于价值的方法**：目标是学习最优动作价值函数 $Q^*(s, a) = \max_\pi Q^\pi(s, a)$ 或最优状态价值函数 $V^*(s)$。策略由价值函数隐式定义，如 $\pi(s) = \arg\max_a Q(s, a)$。学习的核心是贝尔曼最优方程的迭代求解。
+ **基于策略的方法**：直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升最大化 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$。策略梯度定理给出 $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$，其中 $Q^{\pi_\theta}(s,a)$ 的真实值通常用采样回报来估计。
+ **Actor‑Critic 方法**：同时学习参数化策略 $\pi_\theta$（Actor）和价值函数 $V_\phi(s)$ 或 $Q_\phi(s,a)$（Critic）。Actor 使用策略梯度更新，梯度中的 $Q^{\pi_\theta}$ 由 Critic 提供估计；Critic 使用时序差分或蒙特卡洛方法更新。这显著降低了策略梯度的方差，加速收敛。

### 经典方法
#### 1. 基于价值的方法
**（1）Q‑Learning**  
Watkins 和 Dayan (1992) 提出 Q‑Learning，是最经典的异策略基于价值方法。其更新规则利用下一状态的最大 Q 值：  
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right].  
$$
Q‑Learning 直接逼近最优 Q 值，行为策略可与目标策略分离，适用于离线或经验回放 [2]。

**（2）SARSA**  
Rummery 和 Niranjan (1994) 提出 SARSA，是同策略基于价值方法。更新时使用下一状态实际选择的动作：  
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right].  
$$
SARSA 学习当前策略的 Q 值，策略改进与探索绑定，在风险敏感场景下更稳健 [3]。

**（3）DQN（Deep Q‑Network）**  
Mnih 等 (2015) 将 Q‑Learning 与深度神经网络结合，提出 DQN。使用经验回放和目标网络稳定训练，从高维视觉输入直接学习控制策略，在 Atari 游戏上达到人类水平。后续改进包括 Double DQN（van Hasselt 等, 2016）[4]、优先经验回放（Schaul 等, 2016）[5] 等。

#### 2. 基于策略的方法
**（1）REINFORCE**  
Williams (1992) 提出 REINFORCE，是最基础的蒙特卡洛策略梯度方法。梯度估计为：  
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right],  
$$
其中 $G_t$ 是从 $t$ 时刻起的折扣累积回报。REINFORCE 无偏但方差极高，通常需要大量样本才能收敛 [6]。

**（2）TRPO（Trust Region Policy Optimization）**  
Schulman 等 (2015) 提出 TRPO，通过 KL 散度约束限制策略更新步长，保证策略在信任域内单调改进。其优化问题为：  
$$
\max_\theta \mathbb{E}_{s,a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \hat{A}^{\pi_{\text{old}}}(s,a) \right], \quad \text{s.t. } \mathbb{E}_{s \sim \pi_{\text{old}}} [D_{\text{KL}}(\pi_{\text{old}} | \pi_\theta)] \leq \delta.  
$$
TRPO 在理论上保证了稳定改进，但实现复杂 [7]。

#### 3. Actor‑Critic 方法
**（1）A2C/A3C（Advantage Actor‑Critic）**  
Mnih 等 (2016) 提出 A3C，使用优势函数 $A(s,a) = Q(s,a) - V(s)$ 代替原始回报以降低方差。多个并行 Actor 异步与环境交互，更新全局 Critic 网络。其同步版本 A2C 更易实现。A3C 在 Atari 等基准上显著优于当时的基于价值方法 [8]。

**（2）PPO（Proximal Policy Optimization）**  
Schulman 等 (2017) 提出 PPO，通过截断重要性采样比率简化 TRPO：  
$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right],  
$$
其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$。PPO 保留了信任域方法的稳定性，同时实现简单，是深度强化学习中使用最广泛的算法之一 [9]。

**（3）DDPG（Deep Deterministic Policy Gradient）**  
Lillicrap 等 (2016) 提出 DDPG，将 DQN 扩展到连续动作空间的异策略 Actor‑Critic 方法。Actor 输出确定性动作，Critic 使用异策略 TD 学习更新，Actor 通过确定性策略梯度更新。DDPG 在连续控制任务上首次实现了端到端深度强化学习 [10]。

**（4）SAC（Soft Actor‑Critic）**  
Haarnoja 等 (2018) 提出 SAC，在最大熵框架下训练异策略随机策略。其优化目标包含期望回报和策略熵：  
$$
J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t) \sim \pi} [r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))].  
$$
SAC 在连续控制任务上样本效率和稳定性均显著优于此前方法，且对超参数鲁棒 [11]。

### 参考文献
[1] Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.

[2] Watkins, C. J. C. H., & Dayan, P. (1992). Q‑Learning. _Machine Learning_, 8(3‑4), 279–292.

[3] Rummery, G. A., & Niranjan, M. (1994). On‑Line Q‑Learning Using Connectionist Systems. _Technical Report CUED/F‑INFENG/TR 166_, Cambridge University.

[4] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q‑learning. _Proceedings of the 30th AAAI Conference on Artificial Intelligence_, 2094–2100.

[5] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. _Proceedings of the International Conference on Learning Representations (ICLR)_.

[6] Williams, R. J. (1992). Simple Statistical Gradient‑Following Algorithms for Connectionist Reinforcement Learning. _Machine Learning_, 8(3‑4), 229–256.

[7] Schulman, J., Levine, S., Abbeel, P., Jordan, M. I., & Moritz, P. (2015). Trust Region Policy Optimization. _Proceedings of the 32nd International Conference on Machine Learning (ICML)_, 1889–1897.

[8] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. _Proceedings of the 33rd International Conference on Machine Learning (ICML)_, 1928–1937.

[9] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. _arXiv preprint arXiv:1707.06347_.

[10] Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous Control with Deep Reinforcement Learning. _Proceedings of the International Conference on Learning Representations (ICLR)_.

[11] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor‑Critic: Off‑Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. _Proceedings of the 35th International Conference on Machine Learning (ICML)_, 1861–1870.



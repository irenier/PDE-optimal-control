# 偏微分方程的最优控制

## 1. 基本概念

偏微分方程最优控制问题的基本数学框架可表述如下：

$$
\begin{aligned}
\min_{(y,u) \in Y \times U} \quad & J(y,u) \\
\text{s.t.} \quad & e(y,u) = 0, \\
& u \in U_{\mathrm{ad}}, \: y \in Y_{\mathrm{ad}},
\end{aligned}
$$

其中，$Y$ 和 $U$ 为Banach空间，$J: Y \times U \to \mathbb{R}$ 为目标函数。$e(y,u)=0$ 代表偏微分方程或偏微分方程组的约束。$u \in U_{\mathrm{ad}}$ 和 $y \in Y_{\mathrm{ad}}$ 分别表示对控制（control）$u$ 和状态（state）$y$ 的可行集约束。该问题通常被称为最优控制问题（optimal control problem）。

### 1.1. 最优性条件

若 $(\bar{y}, \bar{u})$ 是上述问题的最优解，则存在一个伴随状态（adjoint state）$\tilde{p} \in Z^*$，满足如下的一阶最优性条件：

$$
\begin{aligned}
e(\bar{y}, \bar{u}) & = 0, \\
e_y(\bar{y}, \bar{u})^* \tilde{p} & = J_y(\bar{y}, \bar{u}), \\
\langle J_u(\bar{y}, \bar{u}) - e_u(\bar{y}, \bar{u})^* \tilde{p}, u - \bar{u} \rangle_{U^*, U} & \geq 0, \quad \forall u \in U_{\mathrm{ad}}.
\end{aligned}
$$

该系统包含了状态方程、伴随方程以及一个变分不等式，是求解最优控制问题的理论核心。

## 2. 基于神经网络的最优性条件算法

考虑如下形式的一般非线性最优控制问题：
$$
\begin{array}{cl}
\min_{(y, u) \in Y \times U} & J(y, u) \\
\text{s.t.} & e(y, u) = 0 \\
& u \in U_{\mathrm{ad}}
\end{array}
$$

其最优性条件为：
$$
\begin{aligned}
e(\bar{y}, \bar{u}) & = 0 \\
e_y(\bar{y}, \bar{u})^* \bar{p} & = J_y(\bar{y}, \bar{u}) \\
\left\langle J_u(\bar{y}, \bar{u}) - e_u(\bar{y}, \bar{u})^* \bar{p}, u - \bar{u} \right\rangle_{U^*, U} & \geq 0, \quad \forall u \in U_{\mathrm{ad}}
\end{aligned}
$$

本算法的核心思想是直接利用神经网络对上述最优性条件系统进行求解。特别地，最后一个变分不等式通常可以转化为等式约束，从而得到一个从伴随状态到控制的算子 $S: p \mapsto u$。

我们构建两个神经网络，$y_{\mathrm{NN}}(x, t; \theta_y)$ 和 $p_{\mathrm{NN}}(x, t; \theta_p)$，分别用于拟合状态 $y$ 和伴随状态 $p$。随后，通过算子 $S$ 作用于伴随状态的网络输出，得到控制的神经网络表示：
$$
u_{\mathrm{NN}}(x, t; \theta_p) := S p_{\mathrm{NN}}(x, t; \theta_p)
$$

基于此，我们构造如下的损失函数，其由状态方程和伴随方程的残差平方和构成：
$$
\begin{aligned}
L_1(\theta_y, \theta_p) & := \frac{1}{N_1} \sum_{i=1}^{N_1} \left| e\left[y_{\mathrm{NN}}(x_i, t_i; \theta_y), u_{\mathrm{NN}}(x_i, t_i; \theta_p)\right] \right|^2 \\
L_2(\theta_y, \theta_p) & := \frac{1}{N_2} \sum_{i=1}^{N_2} \left| e_y\left[y_{\mathrm{NN}}(x_i, t_i; \theta_y), u_{\mathrm{NN}}(x_i, t_i; \theta_p)\right]^* p_{\mathrm{NN}}(x_i, t_i; \theta_p) - J_y\left[y_{\mathrm{NN}}(x_i, t_i; \theta_y), u_{\mathrm{NN}}(x_i, t_i; \theta_p)\right] \right|^2 \\
L(\theta_y, \theta_p) & := L_1(\theta_y, \theta_p) + \omega L_2(\theta_y, \theta_p)
\end{aligned}
$$
其中 $\omega$ 是一个权重超参数。通过最小化总损失函数 $L(\theta_y, \theta_p)$，我们可以训练神经网络参数 $\theta_y$ 和 $\theta_p$，从而得到原最优控制问题的数值解。

### 2.1. 算法流程与网络架构

#### 算法流程图

![最优性条件算法](figures/algo.svg)

#### 神经网络架构

![神经网络架构](figures/NN.svg)

## 3. 应用实例一：Burgers方程的分布控制问题

### 3.1. 问题描述

Burgers方程是模拟激波传播和反射的经典模型，在流体力学、非线性声学等领域有广泛应用。其最优控制问题也备受关注。以分布控制为例，其数学形式如下：
$$
\begin{array}{cl}
\min_{y, u} & J(y, u) := \frac{1}{2} \|y - y_d\|_{L^2((0, T); (0,1))}^2 + \frac{\lambda}{2} \|u\|_{L^2((0, T); (0,1))}^2 \\
\text{s.t.} & y_t - \nu y_{xx} + y y_x = u, \quad \text{in } (0,1) \times (0, T), \\
& y(0, t) = y(1, t) = 0, \quad \text{in } (0, T), \\
& y(x, 0) = y_0(x), \quad \text{in } (0,1).
\end{array}
$$

### 3.2. 最优性条件

通过构造Lagrange函数并进行变分，可得如下最优性条件：
$$
\begin{aligned}
-\bar{p}_t - \nu \bar{p}_{xx} - \bar{y} \bar{p}_x & = \bar{y} - y_d, & & \text{in } (0,1) \times (0, T), \\
\bar{p}(0, t) = \bar{p}(1, t) & = 0, & & \text{on } (0, T), \\
\bar{p}(x, T) & = 0, & & \text{in } (0,1), \\
\bar{p} + \lambda \bar{u} & = 0, & & \text{a.e. in } (0,1) \times (0, T).
\end{aligned}
$$

### 3.3. 数值结果

实验参数设定为：$T = 1.0$, $\lambda = 0.05$, $\nu = 0.01$。初始条件 $y_0(x)$ 和目标状态 $y_d(x,t)$ 均为分段函数：
$
y_0(x) = y_d(x,t) =
\begin{cases}
    1, & x \in \left(0, \frac{1}{2}\right) \\
    0, & x \in \left(\frac{1}{2}, 1\right)
\end{cases}
$

#### 训练损失下降曲线

![Burgers方程PINN训练损失](figures/burger-loss.svg)

#### 结果对比

| 描述 | 有限元方法 (FEM) | 物理信息神经网络 (PINN) |
| :--- | :---: | :---: |
| **最优控制** | ![FEM最优控制](figures/burger-control-true.svg) | ![PINN最优控制](figures/burger-control.svg) |
| **最优状态** | ![FEM最优状态](figures/burger-state-true.svg) | ![PINN最优状态](figures/burger-state.svg) |
| **最终时刻状态与目标之差** | ![FEM状态误差](figures/burger-state_error-true.svg) | ![PINN状态误差](figures/burger-state_error.svg) |

## 4. 应用实例二：二维抛物型方程的分布控制问题

### 4.1. 问题描述

考虑一个薄片上的热传导问题，其内部存在一个可控热源，且边界热流为零。控制目标是调节内部热源，使得薄片在指定时间段内的温度分布尽可能接近一个目标函数 $y_d(\mathbf{x}, t)$。记 $y(\mathbf{x}, t)$ 为在时刻 $t$、位置 $\mathbf{x}$ 处的温度，$u(\mathbf{x}, t)$ 为对应的内部热源强度。该问题可建模为如下的最优控制问题：

$$
\begin{array}{rl}
\min_{y, u} \quad & J(y, u) := \frac{1}{2} \|y - y_d\|_{L^2((0, T); \Omega)}^2 + \frac{\lambda}{2} \|u\|_{L^2((0, T); \Omega)}^2 \\
\text{s.t.} \quad & y_t - \Delta y = u, \quad \text{in } \Omega \times (0, T), \\
& \frac{\partial y}{\partial \mathbf{n}} = 0, \quad \text{on } \partial \Omega \times (0, T), \\
& y(\cdot, 0) = 0, \quad \text{in } \Omega.
\end{array}
$$

### 4.2. 最优性条件

该问题的最优解 $(\bar{y}, \bar{u})$ 与对应的伴随状态 $\bar{p}$ 满足以下最优性系统：
$$
\begin{aligned}
-\bar{p}_t - \Delta \bar{p} & = \bar{y} - y_d, & & \text{in } \Omega \times (0, T), \\
\frac{\partial \bar{p}}{\partial \mathbf{n}} & = 0, & & \text{on } \partial \Omega \times (0, T), \\
\bar{p}(\cdot, T) & = 0, & & \text{in } \Omega, \\
\bar{p} + \lambda \bar{u} & = 0, & & \text{a.e. in } \Omega \times (0, T).
\end{aligned}
$$

### 4.3. 数值结果

#### 训练损失下降曲线

![热传导方程PINN训练损失](figures/heat-loss.svg)

#### $t=0.25$ 时刻结果对比

| 描述 | 有限元方法 (FEM) | 物理信息神经网络 (PINN) |
| :--- | :---: | :---: |
| **最优控制** | ![FEM最优控制](figures/heat2d-t25-control-true.svg) | ![PINN最优控制](figures/heat2d-t25-control.svg) |
| **最优状态** | ![FEM最优状态](figures/heat2d-t25-state-true.svg) | ![PINN最优状态](figures/heat2d-t25-state.svg) |
| **状态与目标之差** | ![FEM状态误差](figures/heat2d-t25-state_error-true.svg) | ![PINN状态误差](figures/heat2d-t25-state_error.svg) |

#### $t=0.75$ 时刻结果对比

| 描述 | 有限元方法 (FEM) | 物理信息神经网络 (PINN) |
| :--- | :---: | :---: |
| **最优控制** | ![FEM最优控制](figures/heat2d-t75-control-true.svg) | ![PINN最优控制](figures/heat2d-t75-control.svg) |
| **最优状态** | ![FEM最优状态](figures/heat2d-t75-state-true.svg) | ![PINN最优状态](figures/heat2d-t75-state.svg) |
| **状态与目标之差** | ![FEM状态误差](figures/heat2d-t75-state_error-true.svg) | ![PINN状态误差](figures/heat2d-t75-state_error.svg) |

## 5. 结论

本报告介绍了一种基于神经网络的最优性条件算法，用于求解偏微分方程的最优控制问题。通过对二维热传导方程和Burgers方程的分布控制问题进行数值模拟，并与有限元方法的结果进行对比，验证了该算法的有效性。结果表明，基于物理信息神经网络的方法能够较为精确地求解复杂PDE系统的最优控制，为解决此类问题提供了新途径。

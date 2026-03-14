# 01 - Linnear Regression

> Linear regression is a supervised Machine Learning (ML) algorithm that models the linear relationship between a dependent variable (the target or output) and one or more independent variables (the features or inputs).

## Objective

Given a dataset $D$ containing $n$ entries, each with a set of features $x_1, x_2, \dots, x_d$ and the target variable $y$, our objective is to find a mapping function $h$.

$$h:x\mapsto y \quad \forall\:(x, y) \in D$$

## Notation

When you see a term like $x_j^{(i)}$:

- **The base letter**
  - Letter $x$ represents an input feature.
  - Letter $y$ represents the target variable .
- **The subscript ($j$)**:
  - Indicates the specific feature index (the column in our dataset).
  - Note that $0 \leq j \leq d$.
- **The superscript in parentheses ($(n)$ or $(i)$)**
  - Indicates the specific sample or observation (the row in our dataset).
  - Note that $1 \leq j \leq n$.

### Real-World Example

We define an example for house pricing:

- $x_1$ = Square footage.
- $x_2$ = Square foot price.
- $y$ = Price of the house.

Then:

- $x_2^{(5)}$ represents the square foot price for the fifth house in the dataset.
- The target variable is $y = x_1 \times x_2$.
- We try to find a hypothesis $h(x) \approx x_1 \times x_2$

## Linear Regression Function

We define the linear hypothesis function $h_\theta(x)$. The goal is to find a set of parameters (called weights), denoted as $\theta_0, \theta_1, \dots, \theta_d$, so that the prediction $h_\theta(x)$ is as close to the actual $y$ as possible.

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots\: + \theta_d x_d \approx y$$

For simplicity we can introduce an intercept term $x_0 = 1$, which allows us to write the function as follows:

$$h_\theta(x) = \theta_0  x_0 + \theta_1 x_1 + \dots\: + \theta_d x_d = \sum_{i=0}^d \theta_i \times x_i$$

Or in a vectorized form:

$$h_\theta(x) = \begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_d \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_d \end{bmatrix} = \theta^T \cdot\: x$$

## Cost Function

To measure how far our predictions $h_\theta(x)$ are from the real values $y$ based on our current weights, we define a cost function $J(\theta)$. The training process is entirely about minimizing this cost:

$$\theta \leftarrow \arg\min_\theta J(\theta)$$

A general cost function averages the loss over the entire dataset $D$:

$$J(\theta) = \frac{1}{|D|}\sum_{(x, y)\in D} \text{loss}(y, h_\theta(x))$$

While many loss functions exist, Linear Regression standardly uses the Least Squares approach, also called  Mean Squared Error (MSE):

$$J(\theta) = \frac{1}{2}\sum_{i=1}^n(h_\theta(x^{(i)}) - y^{(i)})^2$$

## Gradient Descent

> Gradient Descent is a optimization algorithm used to find the weights $\theta$ that minimize the cost function $J(\theta)$.

### The Algorithm

1. The weights are initially chosen at random.
$$ \theta \longleftarrow \theta_0$$
2. Then, we iteratively update the weights in the opposite direction of the gradient until we reach a minimum:

$$\theta \longleftarrow \theta - \gamma \times \frac{\partial J}{\partial \theta}$$

### Definitions

- **Simultaneous Update**: All weights ($\theta_1, \theta_2, ...\: \theta_j$) should be updated simultaneously.
- **Global Minimum**: Because the linear hypothesis $h(x)$ makes $J(\theta)$ a convex quadratic function (a bowl shape), there is only one minimum. Therefore, any local minimum is guaranteed to be the absolute (global) minimum.
- **Learning Rate ($\gamma$)**: Determines the step size of each update.
  - If $\gamma$ is too small, convergence is slow.
  - If $\gamma$ is too large, the algorithm may overshoot the minimum and diverge.
- **Stochastic Gradient Descent (SGD)**: calculating the derivative over large datasets is computationally expensive, so we estimate it using a subset (Mini-batch).

### Gradient Breakdown. Partial Derivatives

$$
\begin{aligned}
\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta} \left( \frac{1}{2}\:(h_\theta(x) - y)^2 \right) \\
&= \frac{1}{2} \cdot 2 \: (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta} (h_\theta(x) - y) \\
&= (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta} (\sum_{i=0}^{d} \theta_i x_i - y) \\
&= (h_\theta(x) - y) \cdot x_j\\
\end{aligned}
$$

## Vectorization

To implement this algorithm efficiently in code, we translate the summations and individual updates into matrix operations.

$$
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(n)}
\end{bmatrix}
=
\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \dots & x_d^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \dots & x_d^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(n)} & x_2^{(n)} & \dots & x_d^{(n)}
\end{bmatrix}
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_d
\end{bmatrix}
+
\begin{bmatrix}
\epsilon^{(1)} \\
\epsilon^{(2)} \\
\vdots \\
\epsilon^{(n)}
\end{bmatrix}
$$

The training loop consists of three main steps:

1. **Forward Pass** (Calculating predictions)
2. **Backward-propagation** (Calculating gradients)
3. **Weight Update** (Optimizing parameters)

These steps are repeated until convergence is achieved. Convergence is usually detected by storing the previous step MSE and comparing with the new one. (no improvement equals convergence).

### Step 1 - Forward Pass

We make a prediction with our hypothesis $h_\theta(X)$ by multiplying our feature matrix by our weights:

$$
h_\theta(X) =
\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \dots & x_d^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \dots & x_d^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(n)} & x_2^{(n)} & \dots & x_d^{(n)}
\end{bmatrix}
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_d
\end{bmatrix}
=
\begin{bmatrix}
\hat{y}_0 \\
\hat{y}_1 \\
\vdots \\
\hat{y}_d
\end{bmatrix}
= \hat{Y}
$$

However, when our model makes the prediction $\hat{Y}$, it is rarely perfect. We define the error vector $\vec{e}$ (also known as the residuals) as the difference between our predictions and the actual target values $Y$:

$$\vec{e} = \hat{Y} - Y =
\begin{bmatrix}
\hat{y}^{(1)} - y^{(1)} \\
\hat{y}^{(2)} - y^{(2)} \\
\vdots \\
\hat{y}^{(n)} - y^{(n)}
\end{bmatrix}
=
\begin{bmatrix}
e^{(1)} \\
e^{(2)} \\
\vdots \\
e^{(n)}
\end{bmatrix}$$

### Backward-propagation (Gradients)

Instead of calculating the partial derivative for each weight $\theta_j$ individually using a loop, we can compute the entire gradient vector $\nabla_\theta J$ at once. By multiplying the transposed feature matrix $X^T$ by our error vector $\vec{e}$, we sum the errors scaled by their respective feature values across all $n$ samples simultaneously.

$$
X^T \vec{e} =
\begin{bmatrix}
1 & 1 & \dots & 1 \\
x_1^{(1)} & x_1^{(2)} & \dots & x_1^{(n)} \\
x_2^{(1)} & x_2^{(2)} & \dots & x_2^{(n)} \\
\vdots & \vdots & \ddots & \vdots \\
x_d^{(1)} & x_d^{(2)} & \dots & x_d^{(n)}
\end{bmatrix}
\begin{bmatrix}
e^{(1)} \\
e^{(2)} \\
\vdots \\
e^{(n)}
\end{bmatrix}
$$

To keep the gradients stable regardless of dataset size, we usually average them by dividing by $n$:

$$\nabla_\theta J = \frac{1}{n}\: X^T \vec{e} = \frac{1}{n}\: X^T (\hat{Y} - Y)$$

## Weight Update

Finally, we apply the Gradient Descent rule to all weights simultaneously. We multiply the computed gradient vector by our learning rate $\gamma$ and subtract it from our current weights vector $\theta$:

$$\theta \longleftarrow \theta - \gamma \times \nabla_\theta J$$

---

# A short note on optimizing NNs.

Take a NN parametrized with parameters $ \theta $. 
During training, the parameters are updated using differential information relating the performance obtained to the weights used, i.e. using $\nabla L (\theta) = \sum_{i \in \mathcal{D}} \nabla \ell_i (\theta) $, so that weights are iteratively updated according to:
$$ \theta_{t+1} \leftarrow f (\theta_t, \nabla_t L (\theta_t) ), $$

where $f$ is some function of the (current) weights $ \theta_t $ and gradients $\nabla_t L (\theta_t)$.

For conceptual and computational reason, one typically does not use $\nabla L (\theta) $, and rather relies on $ \sum_{i \in \mathcal{B}} \nabla \ell_i (\theta) $, referred to as the *stochastic gradient* for the mini-batch $\mathcal B \subset \mathcal D: \mathcal B \sim \mathcal D$.
Conceptually, stochastic gradients suffer less from poor initialization than their deterministic counterpart, which proves particularly useful in the context of non-convex optimization.
Computationally, estimating the full gradient requires processing *all* the samples in $ \mathcal D $ through the network at all times, which is simply prohibitive for large-scale datasets, resulting in the computational need to process mini-batches $ \mathcal B \subset \mathcal D: \vert \mathcal B \vert \ll \vert \mathcal D \vert $.
Note how SGD still performs an update for the entire parameter vector $\theta$, although it exclusively relies on limited information regarding $\mathcal D$, in particular using $\mathcal B \subset \mathcal D$.

## [Adam](https://arxiv.org/pdf/1412.6980)

The [infamous Adam](https://x.com/2prime_PKU/status/1948549824594485696) update rule suggests weights should be updated using:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

There are multiple aspects in this update rule.
Together, they make Adam a best-of-both-worlds optimization algorithm when it comes down to combining (1) momentum ($\hat{m}_t $) and (2) adaptive learning rates ($\hat{v}_t$).

### Momentum, or $m_t$

The intuition behind momentum is to reuse previous differential information to improve and stabilize optimization. 
In that, momentum typically smooths the trajectory of more standard SGD by aggregating previous ($\tau<t$) gradients into the timestep-$t$ update.

In practice, by defining a coefficient $\beta$ regulating the relevance of *past* gradients when forming the *current* update, one can derive a modified update rule defined as:

$$
\begin{align*}
\theta_{t+1} &= \theta_t - \eta \cdot m_t , \quad m_t = \beta m_{t-1} + (1-\beta) g_t \\
g_t &= \sum_{k \in \mathcal{B}} \nabla \ell_k (\theta_t)
\end{align*}
$$

This update rule maintains previous gradients relevant according to the parameter $\beta$: for $\beta \to 1$ previous gradients dominate the current gradient estimate, whereas for $\beta \to 0$ the current gradient estimate has the most impact on the parameter update.
Crucially, while momentum naturally accomodates for a possibly time-dependant learning rate $\eta = \eta_t$, it still uses an equal learning rate for all parameters, resulting in the need to perform significant tuning to improve practical performance.
Momentum was first introduced by the Soviet mathematician Polyak in the 1960s.


#### Nesterov Momentum
A popular variant of momentum is Nesterov-accelerated momentum. Differently from Polyak's momentum, Nesterov's acceleration uses the momentum $m_t$ as a coarse approximation for $g_t$, and critically only leverages differential information to adjust said approximation *after* having performed a parameter update. Formally,

$$
\begin{align*}
\theta_{t+1} &= \theta_t - \eta \cdot m_t , \quad m_t = \beta m_{t-1} + (1-\beta) g_t \\
g_t &= \sum_{k \in \mathcal{B}} \nabla \ell_k (\theta_t - \eta \beta m_{t-1})
\end{align*}
$$
Effectively, by using $ \ell_k (\theta_t - \eta \beta m_{t-1}) $ in the parameter update, differential information is employed to perform corrections in the direction of momentum.

### Adaptive Learning Rates, or $v_t$
Momentum proves useful in guaranteeing smoother, more stable optimization routines, embedding inertial information into the optimization by reusing differential information collected earlier in the training process.
However, it tragically suffers from the need to sensitivity to hyper-parameters, including both the learning rate $\eta$ and momentum factor $\beta$.
While hyperparameter tuning is oftentimes simply necessary to have obtain good performance, in the 2010s many works ([AdaGrad, 2011](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf), [Adadelta, 2012](https://arxiv.org/pdf/1212.5701), [RMSProp, 2012](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)) set out to reduce the dependancy of the optimization process on the identification of an optimal learning rate, propsing *adaptive scalers* $v_t$ of a given initial learning rate $\eta$.

Different in their implementation relatively to how they all reuse previous information, AdaGrad, Adadelta and RMSProp all rely on the rather similar conceptual underpinning of normalizing the learning rate $\eta$ per parameter by the scale of the updates received.
In this, the intuition behind the different methods is that parameters that receive updates less often (i.e., parameters which stay closer to their initialized value during training) should---to improve on convergence---use larger stepsizes $\eta$ than parameters which receive updates often during training, which in turn should---to increase stability---be updated less drastically. 
Formally, this intuition results in an update rule like:
$$
\begin{align*}
\theta_{t+1} &= \theta_t - 
\frac{\eta}{v_t} g_t, \\
g_t &= \sum_{k \in \mathcal{B}} \nabla \ell_k (\theta_t)
\end{align*}
$$
where the term $v_t \in \mathbb R^d \ni \theta_t \, \forall t $ is used to scale the learning rate per-parameter $\theta_{t,i} \in \theta_t$.

**AdaGrad** uses the sum of the squared gradients up to $t$ to scale the learning rate.
Formally,
$$
\begin{align*}
v_t &= \operatorname{diag}(G_t)^{\tfrac12}, \implies \theta_{t+1} = \theta_t - 
\eta \operatorname{diag}(G_t)^{-\tfrac 12} \odot g_t, \, \tag{AdaGrad} \\
G_t &= \sum_{i=1}^t g_i g_i^\top = G_{t-1} + g_t g_t^\top
\end{align*}
$$

The matrix $G_t$ serves as an accumulator of the information contained in the updates up to $t$, and in particular it can be understood as a measure of the magnitude of per-parameter update up to $t$.
Indeed, considering the $j$-th parameter in $(\operatorname{diag}G_t)$ is the same as measuring the Root Mean Square (RMS) of the variations that intervened on that very same $j$ up to a multipliticative factor depending on $t$ ($\sqrt(t)$), as it follows from 
$$
\operatorname{RMS}(g_1, g_2 \dots, g_t) = \sqrt{\frac 1t \sum_{i=1}^t g_i^2 } \implies \operatorname{diag}(G_t)^{\tfrac 12} = \sqrt{t} \cdot \operatorname{RMS}(g_1, g_2 \dots, g_t)
$$

By scaling the learning rate for parameter $j$ by $\eta$ by $\sqrt{t} \cdot \operatorname{RMS}((g_1)_j, (g_2)_j, \dots, (g_t)_j)$ one has that, at the same point in training (i.e., for the same $t$), less frequently updated parameters (for which the RMS tends to be smaller) receive larger updates compared to more often updated parameters, for which the RMS of previous gradients is larger.

When computing $ G_t = \sum_{i=1}^t g_i g_i^\top $ all past gradients have similar weight across $1, \dots, t$. 
Because earlier in the training procedure gradients are likely to be large due to initialization, considering all gradients equally may result in an excessive shrinking of the learning rate, hindering performance.
**RMSProp** directly addresses this by maintaining a (soft) receptive field of $\frac{1}{1-\gamma}$ steps only when aggregating incoming gradients, using the update:
$$
G_t = \gamma \cdot G_{t-1} + (1-\gamma) \cdot g_t g_t^\top.
$$
In turn, RMSProp effectively maintains the summation of squared gradients more aligned with the current optimization, mitigating the aforementioned excessive shrinking of the learning rate.
In contrast to AdaGrad, RMSProp is therefore typically less sensitive to poor initialization, and just like AdaGrad it retains the need to define a global learning rate $\eta$ to be scaled down when needed.

> Sidenote: **AdaDelta** is an optimization algorithm that learns without defining a global learning rate. In that, it maintains a running average of the square parameter update, and uses it alongside the RMSProp-like average of square gradients to completely sidestep the need to define a global learning rate $\eta$.

### Bias correction, or $\hat{\bullet}$


## [AdamW]()

## [Muon]()

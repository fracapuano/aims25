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

Adagrad

RMSProp

### Bias correction, or $\hat{\bullet}$

## [AdamW]()

## [Muon]()

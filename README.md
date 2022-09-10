## Optimization for Algorithmic Trading

### Background
Much of the literature on systematic trading focuses on testing different 
models that are used to financial price forecasting. What's wrong with 
forecasting financial time series?

1. Price forecasting doesn't align well with the objective of a trader. As a trader, 
I a less interested in whether the price in the next `n` minutes is going up, down, 
or sideways. What I am truly interested in at any point in time is whether I should 
be long, short, or close out any open positions.

2. Financial time series are highly random. A nonexhaustive list of issues
in creating a unified price forecasting model:

**Issues:**
* Leptokurtic characteristic of price changes.
* Apparent nonstationarity and unstable sample variances over time.
* Discontinuities in prices e.g. jump discontinuities at market open or close
    
**Solutions:**
* Fractional price changes. Price changes are random while fractional price
changes can exhibit serial dependence.
* Use of "trading time". Although price changes are highly leptokurtic, price
changes over transaction or volume based time can be nearly gaussian over 
longer periods.

The goal of this project is to create a modeling framework that aligns with the 
objective of a trader. The included methodology and code can be used for 
optimizing trading systems.

### Concept

$X$ = input matrix of covariates <br>
$Y$ = matrix of target labels with elements $y_i \in \{0,1,2\}$ <br>
$F$ = model approximating $P(Y|X)$ <br>
$F_k$ = model approximating $P(Y_k|X)$ <br>

Assume we have some matrix $X$ of covariates that contain predictive information.
The goal is to create a model that exploits this information to systematically
place trades. Trading can be described using a three class sample space directly 
mapping to the three trading positions of "short", "long", and "closed". The labels 
$Y$, representing these three states, can be generated from fixed duration returns, 
which have no guarantee of optimality, or directly optimizing the labels as part 
of the modeling process. Optimizing the model and target labels while conditioning 
on the inputs can be done using an algorithm that accounts for trade frictions and 
"risk". Where trade frictions account for both expected transaction costs and 
slippage.

Given some risk constraint $r_k$ and the expected trade frictions, the algorithm 
maximizes returns. In this context, "risk" can be a measure of dispersion or 
some approximation of tail risk. The optimal solution for $Y$ given by $Y_k$ can 
change depending on this chosen constraint. For each solution $Y_k$, a model 
$F_k$ can be fitted to approximate the probabilistic relationship between $X$ and 
$Y_k$. Then, the estimated probabilities $P(Y_k|X)$ can be used to evaluate the 
expected performance using time series cross validation. Among the canidate 
$F_k(Y_k,X)$ solutions, there exists an optimal joint solution of a model and 
target labels $F^*(Y^*,X)$ conditioned on the input covariates.

### Code
- **createEnv.sh** - bash script for creating virtual python environment.
- **functions.py** - functions used for trade label optimization and testing.
- **main.ipynb** - python notebook demonstrating usage and methodology.

Within the `functions.py` module, the method `target_optimal` contains the 
algorithm for optimizing the labels. This can be used generate the different
potential solutions given input target constraints. Then, the optimal model 
can be found using time series cross validation.


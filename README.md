## Optimization for Algorithmic Trading

### Background
Much of the literature for systematic trading focuses on testing different 
financial price forecasting models. What's wrong with forecasting financial 
time series?

1. Price forecasting doesn't align well with the objective of a trader. As a trader, 
I'm less interested in whether the price over the next `n` periods is going up, down, 
or sideways. What I'm truly interested in at any given point in time, is whether I
should go long, short, or close out open positions.

3. Financial time series are highly random. A nonexhaustive list of issues
associated with price forecasting financial series includes:

**Issues:**
* Leptokurtic characteristic of price changes.
* Apparent nonstationarity or unstable sample variances across periods.
* Discontinuities in prices e.g. jump discontinuities at market open or close.
    
**Solutions:**
* Fractional price changes. Price changes are random while fractional price
changes can exhibit serial dependence.
* Use of "trading time". Although price changes are highly leptokurtic, price
changes over transaction or volume based windows can be nearly gaussian over 
longer periods.

The goal of this project is to create a modeling framework that aligns with the 
objective of a trader. The included methodology and code can be used for 
optimizing trading systems.

### Concept

$X$ = input matrix of covariates <br>
$Y$ = matrix of target labels with elements $y_i \in \{0,1,2\}$ <br>
$F_k$ = model approximating conditional probability $P(Y_k|X)$ <br>

Assume we have some matrix $X$ of covariates that contain predictive information.
The goal is to create a model that exploits this information to systematically
place trades. Trading can be described using a three class sample space directly 
mapping to the three trading positions of "short", "long", and "closed". The labels 
$Y$, representing these three states, can be generated from fixed duration returns, 
which have no guarantee of optimality, or directly optimizing the labels as part 
of the modeling process. Optimizing the model and target labels while conditioning 
on the inputs can be done using a variant of Kadane's algorithm that accounts for 
trade frictions and "risk". Trade frictions should account for both expected 
transaction costs and slippage.

Given some risk constraint $r_k$ and the expected trade frictions, the algorithm 
maximizes returns. In this context, "risk" can be a measure of dispersion or 
some approximation of tail risk. The optimal labels $Y^{\ast}$ can change depending 
on this chosen risk constraint. For each possible solution $Y_k$ corresponding to 
a risk constraint $r_k$, a model $F_k$ can be fitted to approximate the 
conditional relationship between $X$ and $Y_k$. Then, the estimated probabilities 
$P(Y_k|X)$ can be used to evaluate the expected performance using time series 
cross validation. Among the candidate models $F_k(Y_k,X)$, there exists an optimal 
joint model and target label solution $F^{\ast}(Y^{\ast},X)$.

### Code
- **createEnv.sh** - bash script for creating virtual python environment.
- **backtest.py** - class and methods used for FX trade label optimization and testing.
- **backtest_futures.py** - class and methods for trade label optimization and testing
of futures contracts.
- **main.ipynb** - python notebook demonstrating usage and methodology.

Within the `backtest.py` module, the method `target_optimal` contains the 
algorithm for optimizing trade labels. This can be used to generate different
potential solutions given some input target constraints. Then, the optimal model 
can be found using time series cross validation.

### Setup Git Hooks
1. Within the project repo, run `pre-commit install`.
2. Then run `pre-commit autoupdate`.
3. To run pre-commit git hooks for flake8 and black run use 
`pre-commit run --all-files`.


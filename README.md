## Optimization for Algorithmic Trading

### Background
Much of the literature on systematic trading focuses on testing different 
models that are used to financial price forecasting. What's wrong with 
forecasting financial time series?

1. Price forecasting doesn't align well with the objective of a trader. As a trader, 
I don't care if the price in the next `n` minutes is going up, down, or sideways. 
What I am truly interested in at any point in time is whether I should be long, 
short, or close out any open positions.

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
$y$ = target labels where $y \in \{0,1,2\} <br>
$f$ = model approximating $P(y|X)$ <br>

Assume we have some matrix $X$ of covariates that contain predictive information.
The goal is to create a model that exploits this information to systematically
place trades. Trading can be described using a three class sample space directly 
mapping to the three trading states of "short", "long", and "closed". The labels 
$y$ representing these three states can be generated from fixed duration returns, 
which have no guarantee of optimality, or directly optimizing the labels as part 
of the modeling process. Optimizing the model and target labels while conditioning 
on the inputs can be done using a nontrivial version of Kadane's algorithm that 
accounts for trade frictions and "risk". Trade frictions consist of expected 
transaction costs and slippage.

Given some risk constraint $k$ and expected trading frictions, the algorithm 
maximizes risk adjusted returns. The optimal solution for $y$ given by $y_k$ can 
change depending on this chosen risk constraint. For each solution $y_k$, a model 
$f$ can be fitted to approximate the probabilistic relationship between $X$ and 
$y_k$. Then the estimated probabilies $P(y_k|X)$ can be used to evaluate the 
expected performance using time series cross validation.


TODO: If volume time is used, then volume could be a proxy for risk exposure, maximizing 
risk adjusted returns is difficult. Taking the log makes the two components additive 
with a parameter for risk aversion. What is the risk constraint?

### Algorithm
Actual algorithm for solving this is a slightly more involved version of Kadane's 
algorithm, which is typically used for solving dynamic programming problems.

### Code
- **createEnv.sh** - bash script for creating virtual python environment.
- **functions.py** - functions used for trade label optimization and testing.
- **main.ipynb** - python notebook demonstrating usage and methodology.

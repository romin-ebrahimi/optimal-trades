## Optimization for Algorithmic Trading

### Background
Much of the literature on systematic trading focuses on testing different 
models that are used to financial price forecasting. What's wrong with 
forecasting financial time series?

1. Financial time series are highly random. A nonexhaustive list of reasons
why creating a unified price forecasting model is extremely difficult:

Issues:

    - Leptokurtic characteristic of price changes.
    - Apparent nonstationarity and unstable sample variances over time.
    - Discontinuities in prices e.g. jump discontinuities at market open or close
    
Solutions:

    - Fractional price changes. Price changes are random while fractional price
        changes can exhibit serial dependence.
    - Use of "trading time". Although price changes are highly leptokurtic, price
        changes over transaction or volume based time can be nearly gaussian over 
        longer periods.

2. Price forecasting doesn't align well with the objective of a trader. As a trader, 
I don't care if the price in the next `n` minutes is going up, down, or sideways. 
What I am truly interested in at any point in time is whether I should be long, 
short, or close out any open positions. 

The goal of this project is to create a modeling framework that aligns with the 
objective of a trader. This methodology and the included code can be used for 
optimizing trading systems.

### Concept
- Math and thinking of how to optimize trades. Demonstrate relationship between 
predictive inputs, optimal trades, and the model trying to capture the conditional 
relationship.
- If volume time is used, then volume could be a proxy for risk exposure, maximizing 
risk adjusted returns is difficult. Taking the log makes the two components additive 
with a parameter for risk aversion.

### Algorithm
Actual algorithm for solving this is a slightly more involved version of Kadane's 
algorithm, which is typically used for solving dynamic programming problems.

### Code
- **createEnv.sh** - bash script for creating virtual python environment.
- **functions.py** - functions used for trade label optimization and testing.
- **main.ipynb** - python notebook demonstrating usage and methodology.

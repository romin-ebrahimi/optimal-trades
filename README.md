## Optimize Trades for Algorithmic Trading

### What is risk and what is our objective as a trader? What is wrong with price prediction?

### Meta analysis of papers on google scholar regarding price prediction.

### Math and thinking of how to optimize trades. Demonstrate relationship between predictive 
inputs, optimal trades, and the model trying to capture the conditional relationship.
- If volume time is used, then volume could be a proxy for risk exposure, maximizing risk adjusted
returns is difficult. Taking the log makes the two components additive with a parameter for
risk aversion.

### Actual algorithm for solving this is a slightly more involved version of Kadane's algorithm, which
is typically used for solving dynamic programming problems.

## Contents
- **createEnv.sh** - bash script for creating virtual python environment.
- **functions.py** - functions used for optimization and testing.
- **main.ipynb** - python notebook demonstrating usage.

[![Build Status](https://travis-ci.org/theodrd/project.svg?branch=master)

# A stochastic competition model with heterogeneous firms and environment externalities

This repository contains my project for the course of numerical methods at Sciences Po. The code can be found in the src folder. Tests are in the test folder. It also includes a brief article summarazing the goal, methodology, and results.

### A quick presentation

I wanted tackle firms competition in the presence of environmental externalities, and to do so I use the mean field game theory. I constructed a simple model, in which a continum of heterogeneous firms compete in producing an industrial good. However the production of such good induces "C02" emissions, and firms are constrained by the environment. They cannot emit more than a given threshold. I then solve the model, which yields three constitutive equations, and I simulate it to find a numerical solution to these equations. 

I particularly looked at the evolution of total quantities and prices given different environmental constraints. Hence I wanted to understand how firm adapt to exogeneous environmental constraints. Does it tend to reduce production and increase prices? Do firms invest more in a mitigating technology?

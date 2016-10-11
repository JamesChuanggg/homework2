# Homework2 - Policy Gradient 

## Why we need to normalize the advantages?
In practice it is important to normalize advantage function. One good idea is to “standardize” these returns (e.g. subtract mean, divide by standard deviation) before we plug them into backprop. 
This way we’re always encouraging and discouraging roughly half of the performed actions. Mathematically you can also interpret these tricks as a way of controlling the variance of the policy gradient estimator.

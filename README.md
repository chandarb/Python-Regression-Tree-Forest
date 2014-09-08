Python-Regression-Tree
======================

Python implementation of regression trees. See "Classification and Regression Trees" by Breiman et al. (1984).

The regression_tree_cart.py module contains the functions to grow and use a regression tree given some training data.

football_parserf.py is an example implementation that predicts an NFL player's fantasy points given their statistics from the previous year. The data is stored in football.csv. 

Note that this is likely not the best application of regression trees as it is difficult to assert that a tree based structure best describes a player's fantasy points given the parameters. However, it should provide a useful reference when building other models.

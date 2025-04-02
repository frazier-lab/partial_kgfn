### Bayesian Optimization of Function Networks with Partial Evaluations
This repository contains the code implementations for Bayesian Optimization of Function Networks with Partial Evaluations (pKGFN) and its accelerated version (fast_pKGFN).

The pKGFN algorithm is detailed in the paper "Bayesian Optimization of Function Networks with Partial Evaluations," accepted at [ICML2024](https://proceedings.mlr.press/v235/buathong24a.html). The accelerated version is described in "Fast Bayesian Optimization of Function Networks with Partial Evaluations," available on [ArXiv].

## Brief overview
Bayesian Optimization of Function Networks with Partial Evaluations (pKGFN) is an algorithm variant of Bayesian Optimization (BO) which uses to solve optimization problems whose objective function is expensive-to-evaluate and can be constructed as a directed acyclic graph such that each function node can be quired independently with varying evaluation costs. Methods in this repository, moreover, consider the aformentioned settings with an additional property that evaluating downstream nodes does not require physically obtaining outputs from upsteam nodes, but known ranges for each node output have to be provided. Our proposed method, 'fast_pkgfn' is a faster variant of pKGFN algorithm applicable to solve problems with these settings.

## Contents
This repository contains partial_kgfn and results folders.

- partial_kgfn consists of the following folders and files:
1. acquisition -- acquisition function files
    - FN_realization.py -- an AcquisitionFunction class used to sample a network realization from a function network model
    - full_kgfn.py -- an MCAcquisitionFunction class used to compute the knowledge gradient for function network acquisition function with full evaluations
    - partial_kgfn.py -- an MCAcquisitionFunction class used to compute the knowledge gradient for function network acquisition function with partial evaluations
    - tsfn.py -- an AcquisitionFunction class used to compute the Thompson Sampling acquisition function
2. experiments -- runner files for the two test case problems
    - ackleyS_runner.py -- a main file to run Ackley problem
    - ackmat_runner.py -- a main file to run AckMat problem
    - freesolv3_runner.py -- a main file to run FreeSolv3 problem
    - GPs1_runner.py -- a main file to run GP test problem #1
    - GPs2_runner.py -- a main file to run GP test problem #2
    - manufacturing_runner.py -- a main file to run Manu problem
3. model -- gaussian process model for function network
    - dag.py -- a DAG object
    - decoupled_gp_network.py -- a model class for function network
4. optim -- codes to support acquisition function optimization
    - discrete_kgfn_optim.py -- a file containing optimization function used to solve partial_kgfn acquisition function
5. test_functions -- test problems
    - ack_mat.py -- a SyntheticTestFunction class for AckMat problem
    - ackley_sin.py -- a SyntheticTestFunction class for Ackley problem
    - freesolv3.py -- a SyntheticTestFunction class for FreeSolv3 problem
    - GPs1.py -- a SyntheticTestFunction class for GP test problem #1
    - GPs2.py -- a SyntheticTestFunction class for GP test problem #2
    - manufacter_gp.py -- a SyntheticTestFunction class for manufacturing problem
    - pharmaceutical.py -- a SyntheticTestFunction class for pharma problem
    - freesolv_NN_rep3dim.csv -- a data file to construct a FreeSolv problem
6. utils -- utilities functions
    - construct_obs_set.py -- code for constructing observation set according to the DAG of the problem
    - EIFN_optimize_acqf.py -- code for optimizing EIFN acquisition function 
    - gen_batch_x_fantasies.py -- code for generate X fantasies for discrete acquisition functions including fullKGFN and partialKGFN
    - posterior_mean.py -- code for computing posterior mean of the network final node's output
7. run_one_trial.py -- code for running one trial of Bayesian Optimization

- results folder is created to store saved models and optimization results.

## Example code
run_experiment.ipynb is a notebook used to run a problem. 

- First cell: Call the Ackmat problem

- Second cell: Call the AckMat problem runner where users need to specify the followings:
    - trial -- trial number (int)
    - algo -- algorithm name (str): options are "EI", "KG", "Random", "EIFN", "KGFN", "TSFN", "pKGFN", "fast_pKGFN" (Our method)
    - cost -- evaluation cost configuration (str): This should be in the format of "node1cost_node2cost_node3cost_..._nodeKcost"
    - budget -- BO evaluation budget (int)
    - impose_assump -- a boolean variable indicating if the upstream-downstream restriction is imposed. If "True", to evaluate downstream nodes, its parent nodes' outputs have to be obtained beforehand. 
**To use fast_pKGFN, impose_condition is needed to set to False.**

## Software requirements
The entire codebase is written in python. Package requirements are as follows:
  - python=3.9
  - botorch==0.8.4
  - numpy==1.23.5
  - gpytorch==1.10
  - scipy==1.10.1
  - pandas
  - matplotlib
  - jupyter

The corresponding environment may be created via conda and the provided fast_pKGFN_evn.yml file as follows:
```
  conda env create -f fast_pKGFN_evn.yml
  conda activate fast_pKGFN_evn
```

"""Bayesian mobile-sensor search harness (correlated, censored, exact 1-D EIG)."""
from .core import Params, World, Belief, ACTIONS, step_pose, c_dag, cov_block, corr

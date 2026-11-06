"""rlscalar (public release): DNS of scalar plumes, physics and learned source posteriors, and motion policies.

Modules used by the paper
-------------------------
dns2             2-D forced-turbulence DNS with passive scalar plumes
search.core      episode parameters, adjoint mean field, intermittent likelihood, grid posterior (Belief)
search.dnsworld  DNS-based truth for episodes (sensor reads the recorded scalar, and optionally velocity)
search.episode   task sampling
search.policies  motion policies (one-step infotaxis, zigzag, plume tracing, cast-and-surge)
search.postnet   learned posterior network (GRU + pairwise decoder)
search.seek      path sampling used to fit the corrected observation model
"""

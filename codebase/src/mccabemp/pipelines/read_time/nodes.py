"""
This is a boilerplate pipeline 'read_time'
generated using Kedro 1.0.0
"""

import arviz as az
import pandas as pd
import pymc as pm
import pytensor
pytensor.config.cxx = '/usr/bin/clang++'

from mccabemp.exp.reader_sim.read_time import HyperPrior, Simulator


def generate_simulated_data(
    hyperprior_params: dict, simulator_params: dict, runtime_params: dict
) -> pd.DataFrame:
    hp = {"hyper": HyperPrior(**hyperprior_params)}
    simulator = Simulator(**(simulator_params | hp))
    df = simulator.simulate_random_read_count(**runtime_params)

    return df


def fit_pymc_model(df: pd.DataFrame, sampler_params: dict):
    _df = df.query("observed > 0")

    # Map agent id -> consecutive integers 0 … N-1
    agent_id, agents = pd.factorize(_df["agent"])
    y = _df["observed"].values
    treat = _df["intervention"].values.astype(int)  # Ensure integer for indexing
    N_agents = agents.size
    N_obs = y.size

    coords = {"agent": agents}

    # ------------------------------------------------------------------
    # 2. Build the hierarchical ex-Gaussian model
    # ------------------------------------------------------------------
    with pm.Model(coords=coords) as model:
        # --- Hyper-priors ------------------------------------------------
        # Match your HyperPrior values more closely
        mu_mu = pm.Normal("mu_mu", mu=150.0, sigma=10.0)
        mu_sd = pm.HalfNormal("mu_sd", 10.0)  # Increased to be more flexible

        sigma_scale = pm.HalfNormal("sigma_scale", 20.0)
        tau_shape = pm.HalfNormal("tau_shape", 10.0)
        tau_scale = pm.HalfNormal("tau_scale", 20.0)

        # Impact parameters - use log-normal to ensure positivity
        impact_mu_mu = pm.Normal("impact_mu_mu", 0.0, 0.5)  # log-scale
        impact_mu_sigma = pm.HalfNormal("impact_mu_sigma", 0.5)
        impact_tau_mu = pm.Normal("impact_tau_mu", 0.0, 0.5)  # log-scale
        impact_tau_sigma = pm.HalfNormal("impact_tau_sigma", 0.2)

        # --- Agent-level parameters --------------------------------------
        mu = pm.Normal("mu", mu_mu, mu_sd, dims="agent")
        sigma = pm.HalfNormal("sigma", sigma_scale, dims="agent")
        tau = pm.Gamma("tau", alpha=tau_shape, beta=1 / tau_scale, dims="agent")

        # Use log-normal for impact parameters to ensure positivity
        impact_mu_raw = pm.Normal(
            "impact_mu_raw", impact_mu_mu, impact_mu_sigma, dims="agent"
        )
        impact_mu = pm.Deterministic(
            "impact_mu", pm.math.exp(impact_mu_raw), dims="agent"
        )

        impact_tau_raw = pm.Normal(
            "impact_tau_raw", impact_tau_mu, impact_tau_sigma, dims="agent"
        )
        impact_tau = pm.Deterministic(
            "impact_tau", pm.math.exp(impact_tau_raw), dims="agent"
        )

        # --- Likelihood ---------------------------------------------------
        # The key fix: model the two conditions separately

        # For control condition (intervention=False)
        mu_control = mu[agent_id]
        tau_control = tau[agent_id]

        # For treatment condition (intervention=True)
        mu_treatment = mu[agent_id] * impact_mu[agent_id]
        tau_treatment = tau[agent_id] * impact_tau[agent_id]

        # Use pm.math.switch to select parameters based on treatment
        mu_obs = pm.math.switch(treat, mu_treatment, mu_control)
        tau_obs = pm.math.switch(treat, tau_treatment, tau_control)

        # Add small epsilon to prevent tau from becoming too close to zero
        tau_obs = pm.math.maximum(tau_obs, 1e-6)

        # ExGaussian likelihood
        obs = pm.ExGaussian(  # noqa: F841
            "y_obs",
            mu=mu_obs,
            sigma=sigma[agent_id],  # sigma doesn't change with intervention
            nu=tau_obs,
            observed=y,
        )

        # Sample posterior
        trace = pm.sample(
            **sampler_params
        )

    return trace

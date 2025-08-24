import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    from __future__ import annotations
    import marimo as mo
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    from dataclasses import dataclass
    from typing import Tuple


    @dataclass
    class HyperPrior:
        """Hyper-parameters governing the parent distributions."""

        mu_mu: float = 300.0  # Mean of the normal parent for mu
        mu_sd: float = 30.0  # SD   of the normal parent for mu
        sigma_shape: float = 2.0  # Shape for half-normal
        sigma_scale: float = 30.0
        tau_shape: float = 2.0  # Shape for Gamma prior on tau
        tau_scale: float = 50.0
        mu_impact_mu: float = 0.9
        mu_impact_sigma: float = 0.3
        tau_impact_mu: float = 0.9
        tau_impact_sigma: float = 0.1


    # --------------------------------------------------------------
    # 2. Single agent model
    # --------------------------------------------------------------
    @dataclass
    class Agent:
        """Represents one agent with true latent parameters and data-generation."""

        mu: float
        sigma: float
        tau: float
        impact_mu: float
        impact_tau: float

        def sample_observations(
            self, size: int = 1, impact: bool = False
        ) -> npt.NDArray[np.float64]:
            """Generate size RT observations from the agent's ex-Gaussian."""

            # Ex-Gaussian = Normal + Exponential
            normal_part = np.random.normal(self.mu, self.sigma, size)
            exp_part = np.random.exponential(self.tau, size)

            if impact:
                observed = (normal_part * self.impact_mu) + (
                    exp_part * self.impact_tau
                )
            else:
                observed = normal_part + exp_part

            return observed

        @classmethod
        def from_hyper(cls, hyper: HyperPrior) -> Agent:
            """Draw latent parameters from the hyper-priors."""
            mu = np.random.normal(hyper.mu_mu, hyper.mu_sd)
            sigma = np.random.gamma(hyper.sigma_shape, scale=hyper.sigma_scale)
            tau = np.random.gamma(hyper.tau_shape, scale=hyper.tau_scale)
            impact_mu = np.random.normal(hyper.mu_impact_mu, hyper.mu_impact_sigma)
            impact_tau = np.random.normal(
                hyper.tau_impact_mu, hyper.tau_impact_sigma
            )
            return cls(
                mu=mu,
                sigma=sigma,
                tau=tau,
                impact_mu=impact_mu,
                impact_tau=impact_tau,
            )


    # --------------------------------------------------------------
    # 3. Simulation container
    # --------------------------------------------------------------
    class Simulator:
        """Top-level simulator holding N agents."""

        def __init__(self, N: int, hyper: HyperPrior | None = None):
            self.hyper = hyper or HyperPrior()
            self.agents: Tuple[Agent, ...] = tuple(
                Agent.from_hyper(self.hyper) for _ in range(N)
            )  # noqa: UP006

        def simulate(self, n_trials: int) -> pd.DataFrame:
            """Generate n_trials RTs for every agent."""

            observations = []
            agents = []
            intervention = []

            # --------------------
            # Without Intervention
            # --------------------
            for idx, agent in enumerate(self.agents):
                observations.extend(agent.sample_observations(n_trials))
                agents.extend([idx] * n_trials)
                intervention.extend([False] * n_trials)

            # --------------------
            # With Intervention
            # --------------------
            for idx, agent in enumerate(self.agents):
                observations.extend(
                    agent.sample_observations(n_trials, impact=True)
                )
                agents.extend([idx] * n_trials)
                intervention.extend([False] * n_trials)

            df = pd.DataFrame.from_dict(
                {"agent": agents, "observed": observations}
            )
            return df

        def simulate_random_read_count(
            self, trial_scale=2, trial_shape=700, systematic_volume_factor=1.0
        ) -> pd.DataFrame:
            observations = []
            agents = []
            intervention = []

            # --------------------
            # Without Intervention
            # --------------------
            agent_trials = np.random.gamma(
                trial_scale, trial_shape, size=len(self.agents)
            ).astype(int)
            for idx, (agent, n_trials) in enumerate(
                zip(self.agents, agent_trials)
            ):
                # n_trials = n_trials[0]
                observations.extend(agent.sample_observations(n_trials))
                agents.extend([idx] * n_trials)
                intervention.extend([False] * n_trials)

            # --------------------
            # With Intervention
            # --------------------
            agent_trials = np.random.gamma(
                trial_scale, trial_shape, size=len(self.agents)
            ).astype(int)
            for idx, (agent, n_trials) in enumerate(
                zip(self.agents, agent_trials)
            ):
                # n_trials = n_trials[0]
                n_trials = np.floor(
                    n_trials * (agent.impact_mu * systematic_volume_factor)
                ).astype(int)
                observations.extend(
                    agent.sample_observations(n_trials, impact=True)
                )
                agents.extend([idx] * n_trials)
                intervention.extend([True] * n_trials)

            df = pd.DataFrame.from_dict(
                {
                    "agent": agents,
                    "observed": observations,
                    "intervention": intervention,
                }
            )
            return df

        def params(self) -> dict[str, list[float]]:
            """Return the true latent parameters as plain lists, useful for diagnostics."""
            return {
                "mu": [a.mu for a in self.agents],
                "sigma": [a.sigma for a in self.agents],
                "tau": [a.tau for a in self.agents],
            }
    return Agent, HyperPrior, Simulator, np, pd


@app.cell
def _(np, sns):
    sns.histplot(np.random.gamma(2, 700, size=1000))
    return


@app.cell
def _(HyperPrior):
    hp = HyperPrior(
        mu_mu=150, mu_sd=5, tau_shape=6, tau_scale=10, sigma_scale=10, sigma_shape=2
    )
    hp
    return (hp,)


@app.cell
def _(Agent, hp):
    agent = Agent.from_hyper(hp)
    agent
    return (agent,)


@app.cell
def _(agent):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("fivethirtyeight")
    sns.set_context("notebook")

    _ax = sns.histplot(agent.sample_observations(size=5_000), label="Pre-Intervention")
    _ax = sns.histplot(agent.sample_observations(size=5_000, impact=True), label="Post-Intervention")
    _ax.legend()
    return plt, sns


@app.cell
def _(Simulator, hp):
    simulator = Simulator(N=6, hyper=hp)
    simulator
    return (simulator,)


@app.cell
def _(simulator):
    df = simulator.simulate(1000)
    df
    return (df,)


@app.cell
def _(df, plt, sns):
    _fig, _ax = plt.subplots(figsize=(5, 3))
    sns.histplot(df["observed"])
    return


@app.cell
def _(simulator):
    rand_df = simulator.simulate_random_read_count(systematic_volume_factor=1.2)
    rand_df
    return (rand_df,)


@app.cell
def _(plt, rand_df, sns):
    _fig, _ax = plt.subplots(figsize=(5, 3))
    sns.histplot(x="observed", data=rand_df, hue="intervention")
    return


@app.cell
def _(rand_df):
    import scipy.stats as ss

    ss.ttest_ind(rand_df.query("intervention")["observed"], rand_df.query("~intervention")["observed"])
    return


@app.cell
def _(plt, rand_df, sns):
    _fig, _ax = plt.subplots(figsize=(5, 3))
    sns.barplot(x="intervention", y="observed", data=rand_df, hue="intervention")
    return


@app.cell
def _(np):
    def cohend(d1, d2):
    	# calculate the size of samples
    	n1, n2 = len(d1), len(d2)
    	# calculate the variance of the samples
    	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    	# calculate the pooled standard deviation
    	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    	# calculate the means of the samples
    	u1, u2 = np.mean(d1), np.mean(d2)
    	# calculate the effect size
    	return (u1 - u2) / s
    return (cohend,)


@app.cell
def _(cohend, rand_df):
    cohend(rand_df.query("intervention")["observed"], rand_df.query("~intervention")["observed"])
    return


@app.cell
def _(rand_df):
    import statsmodels.api as sm

    _model = sm.GLM.from_formula("observed ~ 1 + intervention", data=rand_df)
    result = _model.fit()
    result.summary()
    return result, sm


@app.cell
def _(result):
    (result.params.iloc[1] / result.params.iloc[0]) * 100
    return


@app.cell
def _(rand_df, sm):
    _model = sm.GEE.from_formula("observed ~ 1 + intervention", groups="agent", data=rand_df)
    gee_result = _model.fit()
    gee_result.summary()
    return (gee_result,)


@app.cell
def _(gee_result):
    (gee_result.params.iloc[1] / gee_result.params.iloc[0]) * 100
    return


@app.cell
def _(rand_df, sm):
    _model = sm.GEE.from_formula(
        "observed ~ 1 + intervention", 
        groups="agent", 
        data=rand_df.query("observed > 0"),
        family=sm.families.NegativeBinomial(alpha=1.5)
    )
    nb_gee_result = _model.fit()
    nb_gee_result.summary()
    return (nb_gee_result,)


@app.cell
def _(nb_gee_result, np):
    np.exp(nb_gee_result.params.iloc[1])
    return


@app.cell
def _(pd, rand_df):
    import pymc as pm
    import arviz as az

    # ------------------------------------------------------------------
    # 1. Prepare the data
    # ------------------------------------------------------------------

    _df = rand_df.query("observed > 0")

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
        tau = pm.Gamma("tau", alpha=tau_shape, beta=1/tau_scale, dims="agent")

        # Use log-normal for impact parameters to ensure positivity
        impact_mu_raw = pm.Normal("impact_mu_raw", impact_mu_mu, impact_mu_sigma, dims="agent")
        impact_mu = pm.Deterministic("impact_mu", pm.math.exp(impact_mu_raw), dims="agent")

        impact_tau_raw = pm.Normal("impact_tau_raw", impact_tau_mu, impact_tau_sigma, dims="agent")  
        impact_tau = pm.Deterministic("impact_tau", pm.math.exp(impact_tau_raw), dims="agent")

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
        obs = pm.ExGaussian(
            "y_obs",
            mu=mu_obs,
            sigma=sigma[agent_id],  # sigma doesn't change with intervention
            nu=tau_obs,
            observed=y,
        )
    return model, pm


@app.cell
def _(model, pm):
    with model:
        # Start with prior predictive to check model setup
        prior_pred = pm.sample_prior_predictive(samples=500, random_seed=42)

        # Sample posterior
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            random_seed=42,
            target_accept=0.95,  # Higher target_accept for better sampling
            return_inferencedata=True
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

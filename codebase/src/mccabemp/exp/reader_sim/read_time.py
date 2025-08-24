from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


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
        self,
        rng: np.random.Generator,
        size: int = 1,
        impact: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Generate size RT observations from the agent's ex-Gaussian."""

        # Ex-Gaussian = Normal + Exponential
        normal_part = rng.normal(self.mu, self.sigma, size)
        exp_part = rng.exponential(self.tau, size)

        if impact:
            observed = (normal_part * self.impact_mu) + (
                exp_part * self.impact_tau
            )
        else:
            observed = normal_part + exp_part

        return observed

    @classmethod
    def from_hyper(cls, hyper: HyperPrior, rng: np.random.Generator) -> Agent:
        """Draw latent parameters from the hyper-priors."""
        mu = rng.normal(hyper.mu_mu, hyper.mu_sd)
        sigma = rng.gamma(hyper.sigma_shape, scale=hyper.sigma_scale)
        tau = rng.gamma(hyper.tau_shape, scale=hyper.tau_scale)
        impact_mu = rng.normal(hyper.mu_impact_mu, hyper.mu_impact_sigma)
        impact_tau = rng.normal(
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

    def __init__(
        self,
        N: int = 20,
        hyper: HyperPrior | None = None,
        seed: int | None = None,
    ):
        self.hyper = hyper or HyperPrior()
        self.rng = np.random.default_rng(seed)
        self.agents: tuple[Agent, ...] = tuple(
            Agent.from_hyper(self.hyper, self.rng) for _ in range(N)
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
            observations.extend(agent.sample_observations(self.rng, n_trials))
            agents.extend([idx] * n_trials)
            intervention.extend([False] * n_trials)

        # --------------------
        # With Intervention
        # --------------------
        for idx, agent in enumerate(self.agents):
            observations.extend(
                agent.sample_observations(self.rng, n_trials, impact=True)
            )
            agents.extend([idx] * n_trials)
            intervention.extend([False] * n_trials)

        df = pd.DataFrame.from_dict(
            {"agent": agents, "observed": observations}
        )
        return df

    def simulate_random_read_count(
        self, trial_scale=2, trial_shape=700, systematic_volume_factor=2.0
    ) -> pd.DataFrame:
        observations = []
        agents = []
        intervention = []

        # --------------------
        # Without Intervention
        # --------------------
        agent_trials = self.rng.gamma(
            trial_scale, trial_shape, size=len(self.agents)
        ).astype(int)
        for idx, (agent, n_trials) in enumerate(
            zip(self.agents, agent_trials)
        ):
            # n_trials = n_trials[0]
            observations.extend(agent.sample_observations(self.rng, n_trials))
            agents.extend([idx] * n_trials)
            intervention.extend([False] * n_trials)

        # --------------------
        # With Intervention
        # --------------------
        agent_trials = self.rng.gamma(
            trial_scale, trial_shape, size=len(self.agents)
        ).astype(int)
        for idx, (agent, n_trials) in enumerate(
            zip(self.agents, agent_trials)
        ):
            # n_trials = n_trials[0]
            n_trials = np.floor(  # noqa: PLW2901
                n_trials * (agent.impact_mu * systematic_volume_factor)
            ).astype(int)
            observations.extend(
                agent.sample_observations(self.rng, n_trials, impact=True)
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

"""
This is a boilerplate pipeline 'read_time'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import generate_simulated_data, fit_pymc_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            generate_simulated_data,
            inputs=["params:default.hyperprior", "params:default.simulator", "params:default.runtime"],
            outputs="simulated_read_times"
        ),
        Node(
            fit_pymc_model,
            inputs=["simulated_read_times", "params:default.sampler_params"],
            outputs="read_time_idata"
        )
    ])

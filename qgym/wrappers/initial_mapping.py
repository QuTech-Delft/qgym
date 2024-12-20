"""This module contains wrappers for :class:`~qgym.envs.InitialMapping`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from qiskit.transpiler import AnalysisPass, Layout

from qgym.envs.initial_mapping import InitialMappingState
from qgym.templates import AgentWrapper
from qgym.utils.qiskit import Circuit, CircuitLike

if TYPE_CHECKING:
    import networkx as nx
    from stable_baselines3.common.base_class import BaseAlgorithm

    from qgym.envs.initial_mapping import InitialMapping


class AgentMapperWrapper(  # pylint: disable=too-few-public-methods
    AgentWrapper[NDArray[np.int_]]
):
    """Wrap any trained stable baselines 3 agent that inherits from
    :class:`~stable_baselines3.common.base_class.BaseAlgorithm`.

    The wrapper makes sure the agent upholds the Mapper protocol , which is required for
    the qgym benchmarking tools.
    """

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        agent: BaseAlgorithm,
        env: InitialMapping,
        max_steps: int = 1000,
        *,
        use_action_masking: bool = False,
    ) -> None:
        """Init of the :class:`AgentMapperWrapper`.

        Args:
            agent: agent trained on the initial mapping environment.
            env: environment the agent was trained on.
            max_steps: maximum number steps the `agent` can take to compute the mapping.
                If the mapping is not found after `max_steps` steps, the algorithm stops
                and raises an error.
            use_action_masking: If ``True`` it is assumed that action masking was used
                during training. The `env` should then have a `action_masks` method
                and the `predict` method of `agent` should accept the keyword argument
                `"action_masks"`. If ``False`` (default) no action masking is used.
        """
        super().__init__(agent, env, max_steps, use_action_masking=use_action_masking)

    def _prepare_episode(self, circuit: Circuit) -> dict[str, nx.Graph]:
        """Extract the interaction graph from `circuit`."""
        interaction_graph = circuit.get_interaction_graph()
        return {"interaction_graph": interaction_graph}

    def _postprocess_episode(  # pylint: disable=unused-argument
        self, circuit: Circuit
    ) -> NDArray[np.int_]:
        state = cast(
            InitialMappingState,
            self.env._state,  # pylint: disable=protected-access
        )
        if not state.is_done():
            msg = (
                "mapping not found, "
                "the episode was truncated or 'max_steps' was reached"
            )
            raise ValueError(msg)
        return state.mapping

    def compute_mapping(self, circuit: CircuitLike) -> NDArray[np.int_]:
        """Compute a mapping of the `circuit` using the provided `agent` and `env`.

        Alias for ``run``.

        Args:
            circuit: Quantum circuit to map.

        Returns:
            Array of which the index represents a physical qubit, and the value a
            virtual qubit.
        """
        return self.run(circuit)


class QiskitMapperWrapper:
    """Wrap any qiskit mapper (Layout algorithm) such that it becomes compatible with
    the qgym framework. This class wraps the qiskit mapper, such that it  is compatible
    with the qgym Mapper protocol, which is required for the qgym benchmarking tools.
    """

    def __init__(self, qiskit_mapper: AnalysisPass) -> None:
        """Init of the :class:`QiskitMapperWrapper`.

        Args:
            qiskit_mapper: The qiskit mapper (:class:`~qiskit.transpiler.Layout`) to
                wrap.
        """
        self.mapper = qiskit_mapper

    def compute_mapping(self, circuit: CircuitLike) -> NDArray[np.int_]:
        """Compute a mapping of the `circuit` using the provided `qiskit_mapper`.

        Args:
            circuit: Quantum circuit to map.

        Returns:
            Array of which the index represents a physical qubit, and the value a
            virtual qubit.
        """
        circuit = Circuit(circuit)

        self.mapper.run(circuit.dag)
        layout: Layout = self.mapper.property_set["layout"]
        return np.fromiter(
            map(layout.__getitem__, circuit.dag.qubits), int, circuit.dag.num_qubits()
        )

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"{self.__class__.__name__}[{self.mapper}]"

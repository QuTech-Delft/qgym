"""This module contains wrappers for :class:`~qgym.envs.InitialMapping`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Layout

from qgym.utils.qiskit_utils import (
    _get_qreg_to_int_mapping,
    get_interaction_graph,
    parse_circuit,
)

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from qgym.envs import InitialMapping


class AgentMapperWrapper:  # pylint: disable=too-few-public-methods
    """Wrap any trained stable baselines 3 agent that inherits from
    :class:`~stable_baselines3.common.base_class.BaseAlgorithm`.

    The wrapper makes sure the agent upholds the Mapper protocol , which is required for
    the qgym benchmarking tools.
    """

    def __init__(
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
        self.agent = agent
        self.env = env
        self.max_steps = max_steps
        self.use_action_masking = use_action_masking

    def compute_mapping(self, circuit: QuantumCircuit | DAGCircuit) -> NDArray[np.int_]:
        """Compute a mapping of the `circuit` using the provided `agent` and `env`.

        Args:
            circuit: Quantum circuit to map.

        Returns:
            Array of which the index represents a physical qubit, and the value a
            virtual qubit.
        """
        interaction_graph = get_interaction_graph(circuit)
        obs, _ = self.env.reset(options={"interaction_graph": interaction_graph})

        predict_kwargs = {"observation": obs}
        for _ in range(self.max_steps):
            if self.use_action_masking:
                predict_kwargs["action_masks"] = self.env.action_masks()  # type: ignore[attr-defined]

            action, _ = self.agent.predict(**predict_kwargs)  # type: ignore[arg-type]
            predict_kwargs["observation"], _, done, _, _ = self.env.step(action)
            if done:
                break

        if not done:
            msg = (
                "mapping not found, "
                "the episode was truncated or 'max_steps' was reached"
            )
            raise ValueError(msg)

        return obs["mapping"]


class QiskitMapperWrapper:
    """Wrap any qiskit mapper (:class:`~qiskit.transpiler.Layout`) such that it becomes
    compatible with the qgym framework. This class wraps the qiskit mapper, such that it
    is compatible with the qgym Mapper protocol, which is required for the qgym
    benchmarking tools.
    """

    def __init__(self, qiskit_mapper: Layout) -> None:
        """Init of the :class:`QiskitMapperWrapper`.

        Args:
            qiskit_mapper: The qiskit mapper (:class:`~qiskit.transpiler.Layout`) to
                wrap.
        """
        self.mapper = qiskit_mapper

    def compute_mapping(self, circuit: QuantumCircuit | DAGCircuit) -> NDArray[np.int_]:
        """Compute a mapping of the `circuit` using the provided `qiskit_mapper`.

        Args:
            circuit: Quantum circuit to map.

        Returns:
            Array of which the index represents a physical qubit, and the value a
            virtual qubit.
        """
        dag = parse_circuit(circuit)

        self.mapper.run(dag)
        layout = self.mapper.property_set["layout"]

        # Convert qiskit layout to qgym mapping
        qreg_to_int = _get_qreg_to_int_mapping(dag)
        iterable = (qreg_to_int[layout[i]] for i in range(dag.num_qubits()))
        return np.fromiter(iterable, int, dag.num_qubits())

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"{self.__class__.__name__}[{self.mapper}]"

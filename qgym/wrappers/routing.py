"""This module contains wrappers for :class:`~qgym.envs.Routing`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from qgym.utils.qiskit_utils import (
    get_interaction_circuit,
    insert_swaps_in_circuit,
    parse_circuit,
)
from qgym.envs.routing import RoutingState
from qgym.templates.wrappers import AgentWrapper

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from stable_baselines3.common.base_class import BaseAlgorithm
    from qgym.envs.routing import Routing


class AgentRoutingWrapper(AgentWrapper[DAGCircuit]):  # pylint: disable=too-few-public-methods
    """Wrap any trained stable baselines 3 agent that inherits from
    :class:`~stable_baselines3.common.base_class.BaseAlgorithm`.

    The wrapper makes sure the agent upholds the QubitRouting protocol , which is
    required for the qgym benchmarking tools.
    """

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        agent: BaseAlgorithm,
        env: Routing,
        max_steps: int = 1000,
        *,
        use_action_masking: bool = False,
    ) -> None:
        """Init of the :class:`AgentRoutingWrapper`.

        Args:
            agent: agent trained on a qubit routing environment.
            env: environment the agent was trained on.
            max_steps: maximum number steps the `agent` can take to compute the qubit
                routing. If the mapping is not found after `max_steps` steps, the
                algorithm stops and raises an error.
            use_action_masking: If ``True`` it is assumed that action masking was used
                during training. The `env` should then have a `action_masks` method
                and the `predict` method of `agent` should accept the keyword argument
                `"action_masks"`. If ``False`` (default) no action masking is used.
        """
        super().__init__(agent, env, max_steps, use_action_masking=use_action_masking)

    def _prepare_episode(
        self, circuit: QuantumCircuit | DAGCircuit
    ) -> dict[str, NDArray[np.int_]]:
        """Extract the interaction circuit from `circuit`."""
        interaction_circuit = get_interaction_circuit(circuit)
        return {"interaction_circuit": interaction_circuit}

    def _postprocess_episode(self, circuit: DAGCircuit) -> DAGCircuit:
        """Route `circuit` based on the findings of the current episode."""
        state = cast(RoutingState, self.env._state)  # pylint: disable=protected-access
        if not state.is_done():
            msg = (
                "routing not found, "
                "the episode was truncated or 'max_steps' was reached"
            )
            raise ValueError(msg)
        return insert_swaps_in_circuit(circuit, state.swap_gates_inserted)

    def compute_routing(self, circuit: QuantumCircuit | DAGCircuit) -> DAGCircuit:
        """Route the `circuit` using the provided `agent` and `env`.

        Args:
            circuit: Quantum circuit to route. The circuit must be a physical circuit,
                i.e., contain exactly one qubit register with name `"q"` and should
                contain exclusively operations that act on 1 or 2 qubits.

        Returns:
            Routed circuit, i.e. a quantum circuit that only contains two qubit gates
            between qubits that are part of the connection graph.
        """
        return self.run(circuit)


class QiskitRoutingWrapper:
    """Wrap any qiskit Router such that it becomes compatible with the qgym framework.
    This class wraps the qiskit mapper, such that it is compatible with the qgym Routing
    protocol, which is required for the qgym benchmarking tools.
    """

    def __init__(self, qiskit_router: TransformationPass) -> None:
        """Init of the :class:`QiskitRoutingWrapper`.

        Args:
            qiskit_router: The qiskit routing algorithm wrap.
        """
        self.routing = qiskit_router

    def compute_routing(self, circuit: QuantumCircuit | DAGCircuit) -> DAGCircuit:
        """Compute a routed version of the `circuit` using the provided `qiskit_router`.

        Args:
            circuit: Quantum circuit to compute the qubit routing off.

        Returns:
            Routed version of the input circuit.
        """
        dag = parse_circuit(circuit)
        return self.routing.run(dag)

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"{self.__class__.__name__}[{self.routing}]"

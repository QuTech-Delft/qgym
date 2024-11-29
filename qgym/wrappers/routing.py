"""This module contains wrappers for :class:`~qgym.envs.Routing`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from qgym.utils.qiskit_utils import (
    get_interaction_circuit,
    insert_swaps_in_circuit,
    parse_circuit,
)

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from qgym.envs import Routing


class AgentRoutingWrapper:  # pylint: disable=too-few-public-methods
    """Wrap any trained stable baselines 3 agent that inherits from
    :class:`~stable_baselines3.common.base_class.BaseAlgorithm`.

    The wrapper makes sure the agent upholds the QubitRouting protocol , which is
    required for the qgym benchmarking tools.
    """

    def __init__(
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
        self.agent = agent
        self.env = env
        self.max_steps = max_steps
        self.use_action_masking = use_action_masking
        if self.use_action_masking and not hasattr(self.env, "action_masks"):
            msg = "use_action_mask is True, but env has no action_masks attribute"
            raise TypeError(msg)

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
        interaction_circuit = get_interaction_circuit(circuit)
        obs, _ = self.env.reset(options={"interaction_circuit": interaction_circuit})

        predict_kwargs = {"observation": obs}
        for _ in range(self.max_steps):
            if self.use_action_masking:
                action_masks = self.env.action_masks()  # type: ignore[attr-defined]
                predict_kwargs["action_masks"] = action_masks

            action, _ = self.agent.predict(**predict_kwargs)
            predict_kwargs["observation"], _, done, _, _ = self.env.step(action)
            if done:
                break

        if not done:
            msg = (
                "routing not found, "
                "the episode was truncated or 'max_steps' was reached"
            )
            raise ValueError(msg)

        return insert_swaps_in_circuit(
            circuit,
            self.env._state.swap_gates_inserted,  # type: ignore[attr-defined]
        )


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

"""Generic abstract base class for RL agent wrappers.

All agnet wrappers should inherit from ``AgentWrapper``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qgym.utils.qiskit import Circuit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stable_baselines3.common.base_class import BaseAlgorithm

    from qgym.templates.environment import Environment
    from qgym.utils.qiskit import CircuitLike

WrapperOutputT = TypeVar("WrapperOutputT")


class AgentWrapper(  # pylint: disable=too-few-public-methods
    ABC, Generic[WrapperOutputT]
):
    """Wrap any trained stable baselines 3 agent that inherits from
    :class:`~stable_baselines3.common.base_class.BaseAlgorithm`.
    """

    def __init__(
        self,
        agent: BaseAlgorithm,
        env: Environment[Any, Any],
        max_steps: int = 1000,
        *,
        use_action_masking: bool = False,
    ) -> None:
        """Init of the :class:`AgentWrapper`.

        Args:
            agent: agent trained on the provided environment.
            env: environment the agent was trained on.
            max_steps: maximum number steps the `agent` can take to complete an episode.
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

    @abstractmethod
    def _prepare_episode(self, circuit: Circuit) -> Mapping[str, Any]:
        """Prepare the episode options with the information from the provided circuit.

        Args:
            circuit: Quantum circuit to extract the episode information from.

        Returns:
            Mapping containing the options that should be provided to
            ``self.env.reset``.
        """

    def _run_epsiode(self, options: Mapping[str, Any]) -> None:
        """Run an episode with the provided options.

        Args:
            options: Mapping to provide to the options argument of ``self.env.reset``.
        """
        obs, _ = self.env.reset(options=options)

        predict_kwargs = {"observation": obs}
        for _ in range(self.max_steps):
            if self.use_action_masking:
                action_masks = self.env.action_masks()  # type: ignore[attr-defined]
                predict_kwargs["action_masks"] = action_masks

            action, _ = self.agent.predict(**predict_kwargs)
            predict_kwargs["observation"], _, done, _, _ = self.env.step(action)
            if done:
                break

    @abstractmethod
    def _postprocess_episode(self, circuit: Circuit) -> WrapperOutputT:
        """Postprocess the epsiode.

        Extract the useful information from ``self.env`` and do something with it.
        """

    def run(self, circuit: CircuitLike) -> WrapperOutputT:
        """Prepare, run and postprocess an episode.

        Output is based on the provided agent, env and circuit combination.

        Args:
            circuit: Quantum circuit to run the episode for.

        Returns:
            Some useful information extracted from the episode.
        """
        circuit = Circuit(circuit)
        options = self._prepare_episode(circuit)
        self._run_epsiode(options)
        return self._postprocess_episode(circuit)

"""The qgym package consist of gyms and tools used for reinforcement learning (RL)
environments in the Quantum domain. It's main purpose is to easily create RL
environments for the different passes of the OpenQL framework, by simply initializing an
environment class. This abstraction of the environment allows RL developers to develop
RL agents to improve the OpenQL framework, without requiring prior knowledge of OpenQL.


Example:
    We want to create an environment for the OpenQL pass of initial mapping for a system
    with a QPU topology of 3x3. Using the ``qgym`` package this becomes:

    .. code-block:: python

        from qgym.envs import InitialMapping
        env = InitialMapping(0.5, connection_grid_size=(3,3))

    We can then use the environment in the code block above to train a stable baseline
    RL agent using the following code:

    .. code-block:: python

        from stable_baselines3 import PPO
        model = PPO("MultiInputPolicy", env, verbose=1)
        model.learn(int(1e5))

"""

__version__ = "0.1.0-alpha"

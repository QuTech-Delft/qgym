class Environment:
    def __init__(self, initialisation_input) -> None:
        """
        Initialize the action space, state space and initial state

        :param initialisation_input: this could for example be cQASM file
        """
        # main attributes
        self._action_space = None
        self._state_space = None
        self._state = None

        # other attributes
        self.step_nmbr = 0
        # todo: more?

    @property
    def state(self):
        """
        Current state of the environment.
        """
        return self._state

    @property
    def state_space(self):
        """
        Description of the full state space, i.e. collection of all possible states.
        """
        return self.state_space

    @property
    def action_space(self):
        """
        Description of the current action space, i.e. the set of allowed actions (potentially depending on the current
        state.
        """
        return self._action_space

    def step(self, action):
        """
        Compute a new state based on the input action. Also give a reward, done-indicator and (optional) debugging info
        based on the updated state.

        :param action: Valid action to take.
        :return: A tuple containing four entries: 1) The updated state; 2) Reward of the new state; 3) Boolean value
            stating whether the new state is a final state (i.e. if we are done); 4) Additional (debugging) information.
        """
        # Increase the step number
        self.step_nmbr += 1

        # Update the current state
        self._update_state(action)

        # Compute the reward, done indicator and define debugging info
        reward = self._compute_reward()
        done = self._is_done()
        info = {"step_nmbr": self.step_nmbr}

        return self.state, reward, done, info

    def random_action(self):
        """
        Generate a random action from the action space

        :return: Random action.
        """
        action = None  # Generate some random action from the action space
        return action

    def reset(self, optional_arguments):
        """
        Reset the environment to the initial state. To be used after an episode
        is finished. Optionally, one can implement increasingly more difficult
        environments for curriculum learning, or give new problem altogether.

        :param optional_arguments: turn on curriculum learning or give new problem
        :return: State after reset.
        """
        # Reset to the initial state and step number
        self.state = None
        self.step_nmbr = 0

        # Do some other stuff

        return self.state

    def _update_state(self, action) -> None:
        self.state = None  # update state based on the given action

    def _compute_reward(self) -> float:
        """
        Give a reward based on the current state
        """
        reward = 0.0  # compute a reward based on self.state
        return reward

    def _is_done(self) -> bool:
        """
        Give a done-indicator based on the current state
        """
        done = False  # determine if 'done' based on self.state
        return done

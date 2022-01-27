class Env:
# =============================================================================
#     Initialize the action space, state space and initial state
#     Arguments:
#         initialisation_input     this could for example be cQASM file 
#     Returns:
#         Void
# =============================================================================
    def __init__(self, initialisation_input):
        self.action_space = None
        self.state_space = None
        self.state = None
        # Other initializations?
        self.step_nmbr = 0
    
# =============================================================================
#     Produce a new state and give a reward, done-indicator and (optionally) 
#     debugging info based on the given action
#     Arguments:
#         action       valid action from the action space
#     Returns:
#         self.state   new current state
#         reward       reward based on the new state
#         done         Boolean value indicating if the episode is finished
#         info         dictionary  with debugging info
# =============================================================================
    def step(self, action):
        #Increase the step number
        self.step_nmbr += 1
        
        #Update the current state
        self.state = None

        #Calculate the reward, done indicator and debugging info
        reward = self.__reward_function()
        done   = self.__done_function()
        info   = {"step_nmbr" : self.step_nmbr}
        
        return self.state, reward, done, info
    
# =============================================================================
#     Reset the environment  to the initial state. To be used after an episode 
#     is finished. Optionally, one can implement increasingly more difficult 
#     environments for curriculum learning, or give new problem altogether.
#     Arguments:
#         optional_arguments    turn on curriculum learning or give new problem
#     Returns:
#         self.state            initial state after the reset        
# =============================================================================
    def reset(self, optional_arguments):
        # Reset to the initial state and step number
        self.state     = None
        self.step_nmbr = 0
          
        # Do some other stuff
        
        return self.state
    
# =============================================================================
#     Give a reward based on the current state
# =============================================================================
    def __reward_function(self):
        reward = None #make some sort of reward based on self.state
        return reward
    
# =============================================================================
#     Give a done-indicator based on the current state
# =============================================================================    
    def __done_function(self):
        done = True #determine if 'done' based on self.state
        return done
        


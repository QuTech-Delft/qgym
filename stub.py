# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:48:27 2022

@author: lindesgvd
"""
from gym import Env



class mappingEnv(Env):
    """ Initialize actions, observations, space"""
    def __init__(self):
        # What actions can we take?
        self.action_space = None
        
        # What are the possible observations?
        self.observation_space = None
        
        # Is the initial state random or given?
        self.state = None
        
        # Other initializations?
        
    def __reward_function(self, action):
        pass
    
    """ What happens when we take a step, how do we treat actions """
    def step(self, action):
        
        # What is the reward?
        reward = None
        
        # When are we done?
        done = False
        
        # What kind of info do we want to give
        info = {}
        
        return self.state, reward, done, info
    
    """ Used for visualisation"""
    def render(self):
        pass
    """ Reset after training run or epsidode """
    def reset(self):
        # Reset to the initial state
        self.state = None
        
        # Do some other stuff
        
        return self.state

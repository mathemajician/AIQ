
#
# Q(lambda) algorithm for Sutton and Barto page 184
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#

from Agent import Agent
from numpy import zeros, ones
import numpy as np
from random import randint, randrange, random
import sys


class Q_l(Agent):

    def __init__( self, refm, disc_rate, init_Q, Lambda, alpha, epsilon, gamma=0 ):

        Agent.__init__( self, refm, disc_rate )

        self.num_states  = refm.getNumObs() # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells   = refm.getNumObsCells()
        
        self.init_Q  = init_Q
        self.Lambda  = Lambda
        self.epsilon = epsilon
        self.alpha   = alpha

        # if the internal discount rate isn't set, use the environment value
        if gamma == 0:
            self.gamma = disc_rate
        else:
            self.gamma = gamma

        if self.gamma >= 1.0:
            print "Error: Q learning can only handle an internal discount rate ", \
                  "that is below 1.0"
            sys.exit()
            
        self.reset()


    def reset( self ):

        self.state  = 0
        self.action = 0

        self.Q_value = self.init_Q * ones( (self.num_states, self.num_actions) )
        self.E_trace = zeros( (self.num_states, self.num_actions) )
        self.visits  = zeros( (self.num_states) )


    def __str__( self ):
        return "Q_l(" + str(self.init_Q) + "," + str(self.Lambda) + "," \
               + str(self.alpha) + "," + str(self.epsilon) + "," \
               + str(self.gamma) + ")"


    def perceive( self, observations, reward ):

        if len(observations) != self.obs_cells:
            raise NameError("Q_l recieved wrong number of observations!")

        # convert observations into a single number for the new state
        nstate = 0
        for i in range(self.obs_cells):
           nstate = observations[i] * self.obs_symbols**i

        # set up alisas
        Q = self.Q_value
        E = self.E_trace
        gamma = self.gamma

        #print Q

        # find an optimal action according to current Q values
        opt_action = self.random_optimal( Q[nstate] )

        # action selection
        if self.sel_mode == 0:
            # do an epsilon greedy selection
            if random() < self.epsilon:
                naction = randrange(self.num_actions)
            else:
                naction = opt_action
        else:
            # do a softmax selection
            naction = self.soft_max( Q[nstate], self.epsilon )
    
       # update Q values using old state, old action, reward, new state and next action
        delta_Q = reward + gamma*Q[nstate,opt_action] - Q[self.state,self.action]

        E[self.state,self.action] += 1
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                Q[s,a] = Q[s,a] + self.alpha * delta_Q * E[s,a]
                if naction == opt_action:
                    E[s,a] = E[s,a] * gamma * self.Lambda
                else:
                    E[s,a] = 0.0  # reset on exploration

 
        # update the old action and state
        self.state  = nstate
        self.action = naction

        return naction


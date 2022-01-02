
#
# Simple frequence based agent for testing in AIQ
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#

from Agent import Agent

from numpy import zeros
from numpy import ones

import  numpy as np


from random import randint, randrange, random


class Freq(Agent):

    def __init__( self, refm, disc_rate, epsilon ):

        Agent.__init__( self, refm, disc_rate )
        
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells   = refm.getNumObsCells()

        self.epsilon = epsilon
        
        self.reset()


    def reset( self ):

        self.action = 0

        self.total = zeros( (self.num_actions) )
        self.acts  = ones(  (self.num_actions) )


    def __str__( self ):
        return "Freq(" + str(self.epsilon) + ")"


    def perceive( self, observations, reward ):

        if len(observations) != self.obs_cells:
            raise NameError("Q_l recieved wrong number of observations!")

        # convert observations into a single number for the new state
        nstate = 0
        for i in range(self.obs_cells):
           nstate = observations[i] * self.obs_symbols**i

        # set up alisas
        Total = self.total
        Acts  = self.acts

        Total[self.action] += reward
        Acts[self.action] += 1
        
        # find an optimal action according to mean reward for each action
        opt_action = self.random_optimal( Total/Acts )

        # action selection
        if self.sel_mode == 0:
            # do an epsilon greedy selection
            if random() < self.epsilon:
                naction = randrange(self.num_actions)
            else:
                naction = opt_action
        else:
            # do a softmax selection
            naction = self.soft_max( Total/Acts, self.epsilon )

 
        # update the old action
        self.action = naction

        return naction


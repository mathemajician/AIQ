
#
# HLQ(lambda) agent with epsilon greed exploration
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#

from Agent import Agent

from random import randint, randrange, random

from numpy  import zeros
from numpy  import ones

import numpy as np
import sys


class HLQ_l(Agent):

    def __init__( self, refm, disc_rate, sel_mode, init_Q, Lambda, epsilon=0, gamma=0 ):

        Agent.__init__( self, refm, disc_rate )

        self.num_states = refm.getNumObs() # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells   = refm.getNumObsCells()
        self.sel_mode   = sel_mode
        self.init_Q     = init_Q
        self.Lambda     = Lambda
        self.epsilon    = epsilon

        # if the internal discount rate isn't set, use the environment value
        if gamma == 0:
            self.gamma = disc_rate
        else:
            self.gamma = gamma

        if self.gamma >= 1.0:
            print "Error: HLQ learning can only handle an internal discount rate ", \
                  "that is below 1.0"
            sys.exit()

        self.reset()


    def reset( self ):
        
        self.state  = 0
        self.action = 0

        self.Q_value = self.init_Q * ones( (self.num_states, self.num_actions) )
        self.E_trace = zeros( (self.num_states, self.num_actions) )
        self.Visits  =  ones( (self.num_states, self.num_actions) )
        self.beta    = zeros( (self.num_states, self.num_actions) )


    def __str__( self ):
        return "H_l(" + str(self.sel_mode) + "," + \
               str(self.init_Q) + "," + str(self.Lambda) + "," \
               + str(self.epsilon) + "," + str(self.gamma) + ")"


    def perceive( self, observations, reward ):

        if len(observations) != self.obs_cells:
            raise NameError("HLQ_l recieved wrong number of observations!")

        # convert observations into a single number for the new state
        nstate = 0
        for i in range(self.obs_cells):
           nstate = observations[i] * self.obs_symbols**i

        # alias some things to make equations more managable
        gamma = self.gamma;  Lambda = self.Lambda;
        state = self.state;  action = self.action;

        Q = self.Q_value
        E = self.E_trace        
        V = self.Visits        
        B = self.beta

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
        delta_Q = reward + gamma*Q[nstate,opt_action] - Q[state,action]

        E[state,action] += 1
        V[state,action] += 1

        for s in range(self.num_states):
            for a in range(self.num_actions):
                B[s,a] = E[s,a] / (V[nstate,naction] - gamma*E[nstate,naction]) \
                         * (Lambda*V[nstate,naction]+(nstate==s and naction==a)) \
                         / (Lambda*V[s,a]           +(nstate==s and naction==a))


        for s in range(self.num_states):
            for a in range(self.num_actions):
                # don't let V get too low or things can blow up in B above
                if V[s,a] > 1e-100:
                    V[s,a] *= Lambda
                else:
                    V[s,a] = 1e-100

                if naction == opt_action:
                    E[s,a] *= gamma * Lambda
                else:
                    E[s,a] = 0.0  # reset on exploration
                
                Q[s,a] += B[s,a] * delta_Q


        # update the old action and state
        self.state  = nstate
        self.action = naction

        return naction


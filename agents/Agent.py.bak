
#
# Base class for RL agents in AIQ
#
# Remember to add any new agents to the list in __init__
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#

from scipy import zeros
from scipy import exp
from random import randrange
from random import random


class Agent:

    def __init__( self, refm, disc_rate ):
        self.num_obs     = refm.getNumObs()
        self.num_actions = refm.getNumActions()
        self.sel_mode    = 0
        self.disc_rate   = disc_rate

    def __str__( self ):
        raise NameError("You need to override Agent.__str__")


    def reset( self ):
        raise NameError("You need to override Agent.reset!")


    # Perceive a state and a reward.  Return the action to take.
    def perceive( self, new_obs, reward ):
        raise NameError("You need to override Agent.perceive!")


    # return the index of the highest q_value, choosing one
    # of them at random if multiple optimal values exist
    def random_optimal( self, q_values ):

        max_reward = -1e10
        opt_move_count = 1
        opt_action = 0

        for a in range(len(q_values)):
            if q_values[a] > max_reward:
                max_reward = q_values[a]
                opt_move_count = 1
            elif q_values[a] == max_reward:
                opt_move_count += 1

        opt_move_sel = randrange(opt_move_count)
        opt_move_count = 0

        for a in range(self.num_actions):
            if q_values[a] == max_reward:
                if opt_move_count == opt_move_sel:
                    opt_action = a
                    break
                else:
                    opt_move_count += 1

        return opt_action



    # I use epsilon for the temperature
    def soft_max( self, q_values, epsilon ):

        # the tricky thing is the rescaling to stop the exp() from overflowing
        max_val = -1e200
        min_val =  1e200
        total = 0.0
        rand = 0.0
        i = 0

        if epsilon < 1e-4: epsilon = 1e-4

        rescaled_v = zeros( q_values.shape )

        # clip q_values
        #for i in range(q_values.size):
        #    if q_values[i] > 1e100: 
        #        print "Bad q_values ", q_values
        #        q_values[i] = 1e100

        #print q_values.shape, rescaled_v.shape,

        # find max and min values, do this manually to avoid python call
        for i in range(q_values.size):
            rescaled_v[i] = q_values[i]/epsilon;
            if rescaled_v[i] > max_val: max_val = rescaled_v[i]
            if rescaled_v[i] < min_val: min_val = rescaled_v[i]

        if max_val > 1e8:
            print "warning: max_val exceeds 1e8 : ", max_val

        # rescale and clip if needed
        if min_val < -595.0 or max_val > 595.0:
            for i in range(q_values.size):
                rescaled_v[i] = rescaled_v[i] - max_val + 595.0   # exp(700) is max
                if rescaled_v[i] < -595.0: rescaled_v[i] = -595.0 # clip at bottom

        # compute total of exponentials for normalisation
        total = 0.0
        for i in range(q_values.size): total += exp(rescaled_v[i])
        if total == 0.0: total = 1e-20

        # finally do the random selection
        rand = random()

        for i in range(q_values.size):
            rand -= exp(rescaled_v[i])/total
            if rand < 0.0: break

        return i


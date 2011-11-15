#
# Wrapper for MC-AIXI v1.0 agent by Joel Veness
# The executable file is assumed to be called mc-aixi
# and to be located in the agents directory.
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#

from Agent import Agent
from numpy import zeros, ones, ceil
import  numpy as np
import subprocess
import os
from math import log
from random import randint, randrange, random


# binary string of a given length for an integer n
def bit_str( n, length ):
    digits = [i for i in range(length-1,-1,-1)]
    return ''.join([str(n >> x & 1) for x in digits])

# convert a positive binary string into an integer
def binstr_2_int( bit_s ):
    n = 0
    val = 2**(len(bit_s)-1)
    for i in range(len(bit_s)):
        n += val * int(bit_s[i])
        val /= 2
    return n


class MC_AIXI(Agent):

    def __init__( self, refm, disc_rate, sims, depth, horizon, \
                  epsilon=0.05, threads=1, memory=32 ):

        Agent.__init__( self, refm, disc_rate )

        if epsilon > 1.0: epsilon = 1.0
        if epsilon < 0.0: epsilon = 0.0

        self.refm = refm
        self.sims = int(sims)
        self.depth = int(depth)
        self.horizon = int(horizon)
        self.memory = int(memory)
        self.epsilon = epsilon
        self.threads = int(threads)

        self.obs_cells   = refm.getNumObsCells()
        self.obs_symbols = refm.getNumObsSyms()

        self.obs_bits = int(ceil(log( refm.getNumObs(), 2.0 )))
        self.reward_bits = int(ceil(log( refm.getNumRewards(), 2.0 )))
        self.num_actions = refm.getNumActions()

        print "obs_bits = ", self.obs_bits
        print "reward_bits = ", self.reward_bits

        self.agent = None

        self.reset()

    def __str__( self ):
        return "MC_AIXI(" + str(self.sims) + "," + str(self.depth) + "," \
               + str(self.horizon) + "," + str(self.epsilon) + ")"


    def __del__( self ):
        self.agent.terminate()
        self.ain.close()
        self.aout.close()

            
    def reset( self ):

        # if there is an old agent dispose of it
        if self.agent != None:
            self.agent.terminate()
            self.ain.close()
            self.aout.close()

        # create new agent
        self.agent = subprocess.Popen(["./agents/mc-aixi",
                    "--mc-simulations",   str(self.sims),
                    "--ct-depth",         str(self.depth),
                    "--agent-horizon",    str(self.horizon),
                    "--memsearch",        str(self.memory),
                    "--exploration",      str(self.epsilon),              
		    "--threads",          str(self.threads),
                    "--observation-bits", str(self.obs_bits),
                    "--reward-bits",      str(self.reward_bits),
                    "--agent-actions",    str(self.num_actions)],
		    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        
        self.aout = self.agent.stdout
        self.ain = self.agent.stdin

 

    def perceive( self, obs, reward ):

        if len(obs) != self.obs_cells:
            raise NameError("MC-AIXI recieved wrong number of observations!")

        # convert observations into a single number for the new state
        obs_num = 0
        for i in range(self.obs_cells):
           obs_num = obs[i] * self.obs_symbols**i

        # now convert this number into bits
        obs_string = bit_str( obs_num, self.obs_bits )

        int_reward = int(round((reward+100.0)/200.0*(self.refm.getNumRewards()-1.0)))
        reward_string = bit_str( int_reward, self.reward_bits )

        self.ain.write( obs_string + reward_string + "\n" )
        self.ain.flush()
        
        action = binstr_2_int(self.aout.readline().decode().strip()) 

        return action


#
# A simple agent that is controlled by the user from the keyboard
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3

import sys

from random  import randint
from Agent   import Agent

import random

MANUAL = 0
RANDOM = 1
SAME   = 2

class Manual(Agent):

    def __init__( self, refm, disc_rate ):
        Agent.__init__( self, refm, disc_rate )

        if self.num_actions > 10:
            print "Error: Manual agent only handles 10 or less actions!"
            sys.exit()

        self.mode = MANUAL
        self.last_val = 0
            

    def __str__( self ):
        return "Manual()"


    def reset( self ):
        print
        print "Reset!"
        print


    def perceive( self, obs, reward ):
        print " obs = " + str(obs) + " reward = " + str(reward )

        if self.mode == MANUAL:
            choice = raw_input(" action [0-" + str(self.num_actions-1) + "]  ")[0]
            if choice == "r":
                self.mode = RANDOM
            elif choice == "s":
                self.mode = SAME
            else:
                action = int(choice)
                
        if self.mode == RANDOM: action = random.randint(0, self.num_actions-1)

        if self.mode == SAME: action = self.last_value

        self.last_value = action
        
        return action

    

#
# Trivial agent that takes random actions
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#

from random  import randint
from Agent   import Agent

class Random(Agent):

    def __init__( self, refm, disc_rate ):
        Agent.__init__( self, refm, disc_rate )

    def __str__( self ):
        return "Random()"

    def reset( self ):
        pass

    def perceive( self, obs, reward ):
        return randint( 0, self.num_actions-1 )

    

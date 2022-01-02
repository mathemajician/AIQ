
#
# Base class for reference machines in AIQ
#
# Remember to add any new reference machines to the list in __init__
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3


class ReferenceMachine:

    def __init__( self ):

        self.num_obs         = 0
        self.num_rewards     = 0
        self.num_actions     = 0
        self.obs_symbols     = 0
        self.obs_cells       = 0

    # used for naming log files etc.
    def __str__( self ):
        print "You need to override ReferenceMachine.__str__"

    def getNumObs( self ):
        return self.num_obs

    def getNumRewards( self ):
        return self.num_rewards

    def getNumActions( self ):
        return self.num_actions

    def getNumObsSyms( self ):
        return self.obs_symbols

    def getNumObsCells( self ):
        return self.obs_cells

    # reset environment and return initial state and reward
    def reset( self, program="" ):
        print "You need to override ReferenceMachine.reset!"

    # Perform this action in the current state.  Return the reward and new state.
    def act( self, action ):
        print "You need to override ReferenceMachine.act!"

        

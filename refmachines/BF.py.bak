
#
# BF derived reference machine for use with AIQ
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3


import random
import sys

from ReferenceMachine import *

from numpy import zeros, ones, array, linspace
from scipy import stats, floor, sqrt
from string import replace


INSTRUCTIONS = ['<','>','+','-',',','.','[',']','#', '%' ]


class BF(ReferenceMachine):


    # create a new BF reference machine, default to a tape with 5 symbols
    def __init__( self, num_symbols=5, obs_cells=1, reverse_output=0 ):

        self.num_symbols  = int(num_symbols)
        self.obs_cells    = int(obs_cells)

        if self.num_symbols < 2:
            raise NameError("Error: BF needs at least 2 tape symbols")

        self.obs_symbols = self.num_symbols
        self.reverse_output = (reverse_output == 1)

        # when num_symbols is odd the following gives us a reward balanced ref machine,
        # otherwise it's slightly unbalanced due to the loop exit condition favouring
        # one output reward value over another
        self.mid_symbol = int((self.num_symbols-1)/2) 

        self.action_cells = 1          # hard coded for now in various places
        self.reward_cells = 1          # hard coded for now in various places
        
        self.num_actions = self.num_symbols**self.action_cells
        self.num_obs     = self.obs_symbols**self.obs_cells
        self.num_rewards = self.num_symbols**self.reward_cells

        self.work_tape_len   = 100000
        self.input_tape_len  = 25 * self.action_cells  # contains action history
        self.output_tape_len = self.reward_cells + self.obs_cells # only this cycle

        self.max_steps   = 1000 # limit on how long to execute for in a single cycle

        self.program     = ""

        self.init_machine()


    # this is used by AIQ to name output log files correctly
    def __str__( self ):
        if self.obs_cells == 1 and not self.reverse_output:
            return "BF(" + str(self.num_symbols) + ")"
        elif not self.reverse_output:
            return "BF(" + str(self.num_symbols) + "," + str(self.obs_cells) + ")"
        else:
            return "BF(" + str(self.num_symbols) + "," + str(self.obs_cells) + ",1)"

    # reset the ref machine before a new run and return the first reward and obs
    def reset( self, program="" ):
        self.program = program
        self.init_machine()
        return (0.0, [self.mid_symbol]*self.obs_cells)


    # initialise the machine by clearing the tape and reseting the pointers
    def init_machine( self ):

        x = self.mid_symbol
        self.work_tape    = [x]*self.work_tape_len   # two way, read & write work tape
        self.input_tape   = [x]*self.input_tape_len  # one way, read only input tape
        self.output_tape  = [x]*self.output_tape_len # one way, write only output tape

        self.work_ptr = 0
        self.input_ptr = 0
        self.output_ptr = 0


    # entry point for an agent acting on the environment
    def act( self, action ):

        if action < 0 or action >= self.num_actions:
            raise NameError('invalid action! ' + str(action))

        self.load_input( [action] )
        steps = self.compute( self.program )
        reward, observations = self.get_output()

        return reward, observations, steps


    #
    # all the remaining methods are internal methods specific to the BF ref machine
    #


    # move the old data in the input tape along and load in the new data
    def load_input( self, input_data ):

        if len(input_data) != self.action_cells:
            raise NameError("Error: Input data to environment has wrong length")

        self.input_tape[self.action_cells:] = self.input_tape[:-self.action_cells]
        self.input_tape[:self.action_cells] = input_data


    # read the output tape and convert into reward and action values
    def get_output( self ):
        mid_point = (self.num_symbols-1.0)/2.0

        if self.reverse_output:
            return 100.0*(self.output_tape[self.obs_cells]-mid_point)/mid_point, \
                   self.output_tape[:self.obs_cells]
        else:
            return 100.0*(self.output_tape[0]-mid_point)/mid_point, \
                   self.output_tape[1:]


    # pull out the code between matching [ and ], keeping the last ] as the end of loop
    def extract_loop( self, instructions ):

        level = 1
        i = 1
        length = len(instructions)-1

        while level > 0 and i <= length:
            if  instructions[i] == '[':
                level += 1
            elif instructions[i] == ']':
                level -= 1
            i += 1

        return instructions[1:i]


    # compute a block of instructions, either a whole program or a loop within a program
    def compute( self, program, level = 0 ):

        # only at start of computing the cycle, reset some stuff
        if level == 0:
            self.step = 0
            self.input_ptr  = 0
            self.output_ptr = 0
            self.cycle_end = False

        instr_ptr = 0

        while not self.cycle_end:

            if self.step >= self.max_steps:
                self.cycle_end = True
                break

            if self.output_ptr == self.output_tape_len:
                self.cycle_end = True
                break

            self.step += 1
            instr = program[instr_ptr]

            if   instr == '<':
                self.work_ptr -= 1
                if self.work_ptr < -self.work_tape_len:
                    self.work_ptr = 0 # pointer wrap around

            elif instr == '>':
                self.work_ptr += 1
                if self.work_ptr >= self.work_tape_len:
                    self.work_ptr = 0 # pointer wrap around

            elif instr == '+':
                self.work_tape[self.work_ptr] += 1
                if self.work_tape[self.work_ptr] >= self.num_symbols: # symbol wrap around
                    self.work_tape[self.work_ptr] = 0

            elif instr == '-':
                self.work_tape[self.work_ptr] -= 1
                if self.work_tape[self.work_ptr] < 0:                 # symbol wrap around
                    self.work_tape[self.work_ptr] = self.num_symbols-1

            elif instr == '.':
                self.output_tape[self.output_ptr] = self.work_tape[self.work_ptr]
                self.output_ptr += 1

            elif instr == ',':
                if self.input_ptr >= self.input_tape_len-1:
                     # if reading past history end
                    self.work_tape[self.work_ptr] = self.mid_symbol
                else:
                    self.work_tape[self.work_ptr] = self.input_tape[self.input_ptr]
                    self.input_ptr += 1

            elif instr == '%':
                self.work_tape[self.work_ptr] = random.randrange(self.num_symbols)

            elif instr == '[':

                # pull out loop instructions
                loop = self.extract_loop( program[instr_ptr:] )

                # run the loop
                while self.work_tape[self.work_ptr] != self.mid_symbol \
                          and not self.cycle_end:
                    self.compute( loop, level+1 )

                # jump past loop end including the ]
                instr_ptr += len(loop)

            elif instr == ']':
                return # exit running the loop

            elif instr == '#':
                self.cycle_end = True

            else:
                print "Error: Unknown instruction ", instr
                self.cycle_end = True

            instr_ptr += 1

        return self.step



    # sample a random program, used by BF_sampler.py
    def random_program( self ):

        program = ""
        loop_depth = 0

        while loop_depth >= 0:
            instr = random.choice(INSTRUCTIONS)
            if instr == '#': instr = ']'     # treat # as loop termination
            if instr == '[': loop_depth += 1
            if instr == ']': loop_depth -= 1
            if loop_depth < 0: instr = '#'   # if ] unmatched, end the program
            program += instr

        # remove some simple pointless instruction combinations
        program = replace(program,'+-','')
        program = replace(program,'-+','')
        program = replace(program,'<>','')
        program = replace(program,'><','')
        program = replace(program,'[]','')

        return program





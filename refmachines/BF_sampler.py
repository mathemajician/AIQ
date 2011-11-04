
#
# Software to generate BF program samples and classify them into strata
# for use by the AIQ stratified estimation algorithm.
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3


import random
from numpy import zeros, ones, array
from scipy import linspace, stats, floor, sqrt
from string import replace, lower
import getopt, sys
from os.path import isfile

import BF


STRATA = 21


# get a random program, excluding over time and passive ones
def active_program( refm ):

    program = refm.random_program()
    env_class = test_class( refm, program )
    while env_class == -1 or env_class == 0:
        program = refm.random_program()
        env_class = test_class( refm, program )
        
    return program, env_class


# Test an environment 4 times to determine its class with higher probability

def test_class( refm, program ):

    # must be passive as it lacks read and/or write
    if program.count('.') == 0 or program.count(',') == 0: return 0

    cycles = 200 # cycles to run test for

    s1, r1 = _test_class( refm, cycles, program )
    s2, r2 = _test_class( refm, cycles, program )
    s3, r3 = _test_class( refm, cycles, program )
    s4, r4 = _test_class( refm, cycles, program )
    s5, r5 = _test_class( refm, cycles, program )

    #print program

    # one "over time" puts it in the "over time" class
    if s1 == -1 or s2 == -1 or s3 == -1 or s4 == -1 or s5 == -1:
        #print "found overtime ", s1, s2, s3, s4, s5
        return -1

    # same rewards on each run (excluding first 5 cycles)
    # puts it in the "passive" class
    if  all(r1[4:] == r2[4:])\
    and all(r1[4:] == r3[4:])\
    and all(r1[4:] == r4[4:])\
    and all(r1[4:] == r5[4:]): return 0
        
        
    if s1 == s2 == s3 == s4 == s5 != 99:  # if all agree and not "other"
        return s1
    else:
        # must be "other", so stratify by program length
        l = len(program)
        if   l <  8: env_type = 11
        elif l < 10: env_type = 12
        elif l < 13: env_type = 13
        elif l < 16: env_type = 14
        elif l < 20: env_type = 15
        elif l < 24: env_type = 16
        elif l < 31: env_type = 17
        elif l < 43: env_type = 18
        elif l < 60: env_type = 19
        else:
            env_type = 20
            #print " type 20 ", s1, s2, s3, s4, s5
        return env_type

# test to see if a program falls into various classes by running it
# against a random agent and seeing what happens in one trial.
# This isn't very reliable, so use test_class intead which takes
# several trials.

def _test_class( refm, cycles, program ):

    refm.init_machine()

    env_overtime = False
    
    env_copy     = True
    env_1back    = True
    env_2back    = True
    env_3back    = True
    env_inc      = True
    env_dec      = True
    env_1backinc = True
    env_1backdec = True
    env_cp_ex    = True
    env_1back_ex = True

    env_type = 0

    ooa = 0
    oa = 0
    o_r = 0

    rewards = zeros( (cycles) )
    
    for i in range( cycles ):

        # run with random actions
        input_data = [random.randrange(refm.num_symbols) \
                      for j in range(refm.action_cells)]
        refm.load_input( input_data )
        steps = refm.compute( program )

        #print input_tape[:INPUT_LENGTH], " ", output_tape[0], output_tape[1:], work_tape[0:5]

        a = refm.input_tape[0]

        if not refm.reverse_output:
            r = refm.output_tape[0]
        else:
            r = refm.output_tape[refm.obs_cells]

        rewards[i] = r

        if steps == refm.max_steps:
            env_overtime = True
            break

        # ignore the first 5 cycles as they tend to have startup junk in them
        if i > 5:
            if r != a:                            env_copy     = False
            if r != oa:                           env_1back    = False
            if r != ooa:                          env_2back    = False
            if r != oooa:                         env_3back    = False
            if r != (a+1)%refm.num_symbols:       env_inc      = False
            if r != (a-1)%refm.num_symbols:       env_dec      = False
            if r != (oa+1)%refm.num_symbols:      env_1backinc = False
            if r != (oa-1)%refm.num_symbols:      env_1backdec = False
            if r != a and a != refm.mid_symbol:   env_cp_ex    = False
            if r != oa and oa != refm.mid_symbol: env_1back_ex = False

        oooa = ooa
        ooa = oa
        oa = a
        o_r = r

    if env_overtime:
        env_type = -1
    else:
        if   env_copy:     env_type =  1
        elif env_1back:    env_type =  2
        elif env_2back:    env_type =  3
        elif env_3back:    env_type =  4
        elif env_inc:      env_type =  5
        elif env_dec:      env_type =  6
        elif env_1backinc: env_type =  7
        elif env_1backdec: env_type =  8 
        elif env_cp_ex:    env_type =  9 
        elif env_1back_ex: env_type = 10
        else:
            env_type = 99

    return env_type, rewards



def usage():
    print
    print "AIQ program sample classifier"
    print
    print "python BF_sampler.py -s sample_size -r ref_machine[,para1[,para2[...]]]"
    print

    

def main():

    print
    print "BF reference machine program sampler"
    print

    sample_size = 0
    refm_str = None
    refm_params = []

    # get the command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:r:", ["help"])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    # exit on no arguments
    if opts == []:
        usage()
        sys.exit()

    # parse arguments
    for opt, arg in opts:
        if   opt == "-s": sample_size = int(arg)
        elif opt == "-r": 
            args = arg.split(",")
            refm_str = args.pop(0)
            for a in args:
                refm_params.append( float(a) )
        else:
            print "Unrecognised option"
            usage()
            sys.exit()

    if sample_size == 0:
        print "Error: No sample size set"
        sys.exit()

    if refm_str == None:
        print "Missing reference machine"
        sys.exit()

    if refm_str != "BF":
        print "Can only handle BF reference machine at the moment!"
        sys.exit()

    refm_call = refm_str + "." + refm_str + "("
    
    if len(refm_params) > 0:
        param = refm_params.pop(0)
        refm_call += str(int(param))
        #file_name += str(int(param))
    for param in refm_params: refm_call += "," + str(int(param))
    refm_call += ")"

    # create reference machine
    refm = eval( refm_call )

    # output filename
    file_name = "./samples/"
    file_name += refm_call.partition('.')[2] # strip off the module name and dot
    file_name += ".samples"

    print "Output filename: " + file_name
    print

    # check for existing sample file
    if isfile( file_name ):
        print "Output sample file already exists, do you want to:"
        choice = lower(raw_input(" Append, Overwrite or Quit [a/o/q] ? "))
        if   choice == 'a': mode = 'a'
        elif choice == 'o': mode = 'w'
        else: sys.exit()
    else:
        mode = 'w'

    sample_file = open( file_name, mode )
    

    # generate the samples
    for i in range( sample_size ):
        program, s = active_program( refm )
        sample_file.write( str(s) + " " + program + "\n" )
        sample_file.flush()

    sample_file.close()


if __name__ == "__main__":
    main()

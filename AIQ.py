#!/usr/bin/python
#
# This estimates that Algorithmic Intelligence Quotient (AIQ) of a
# reinforcement learning agent by testing its performance in randomly
# sampled environments with respect to some given reference machine.
#
# Copyright Shane Legg 2011
# Released under the GNU GPLv3
#

from refmachines import *
from agents import *
from math import isnan
from time import sleep, localtime, strftime
from scipy import ones, zeros, floor, array, sqrt, log, ceil, cov
from random import choice
from multiprocessing import Pool

import getopt, sys


# Test an agent by performing both positive and negative reward runs in order
# to get antithetic variance reduction. 
def test_agent( refm_call, a_call, episode_length, disc_rate, stratum, program ):

    # run twice with flipped reward second time
    s1, r1 = _test_agent(refm_call, a_call,  1.0, episode_length,\
                         disc_rate, stratum, program)
    s2, r2 = _test_agent(refm_call, a_call, -1.0, episode_length, \
                         disc_rate, stratum, program)

    # log successful result to file
    if logging and not isnan(r1) and not isnan(r2):
        log_file.write( strftime("%Y_%m%d_%H:%M:%S ",localtime()) \
              + str(s1) + " " + str(r1) + " " + str(r2) + "\n" )
        log_file.flush()
        
    return (s1,r1,r2)


# Perform a single run of an agent in an enviornment and collect the results
def _test_agent( refm_call, agent_call, rflip, episode_length, \
                 disc_rate, stratum, program ):

    # create reference machine
    refm = eval( refm_call )

    # create agent
    agent = eval( agent_call )
    agent.reset()

    disc_reward = 0.0
    discount    = 1.0

    reward, observations = refm.reset( program )

    for i in range( episode_length ):
        action = agent.perceive( observations, rflip*reward )
        reward, observations, steps = refm.act( action )

        # we signal failure with a NaN so as not to upset
        # the parallel map running this with an exception
        if steps == refm.max_steps: return (stratum,float('nan'))

        disc_reward += discount*rflip*reward
        discount    *= disc_rate

    # if discounting normalise (and thus correct for missing tail)
    if disc_rate != 1.0:
        disc_reward /= ( (1.0-disc_rate**(episode_length+1))/(1.0-disc_rate) )
    else:
	    # otherwise just normalise by the episode length
        disc_reward /= episode_length

    # dispose of agent and reference machine
    agent = None
    refm  = None
    
    return stratum, disc_reward



# Simple MC estimator, useful for checking the more complex adaptive estimator.
# It doesn't do logging as the log file assumes dual runs for antithetic variables.
def simple_mc_estimator( refm_call, agent_call, episode_length, disc_rate, \
                         sample_size ):

    print
    result = zeros((len(sample_data)))
    i = 0
    for stratum, program in sample_data:
        rflip = choice([-1,1])
        perf = _test_agent( refm_call, agent_call, rflip, episode_length, disc_rate, \
                            stratum, program )[1]
        if not isnan(perf):
            result[i] = perf
            if i%10 == 0 and i > 10:
                mean = result[:i].mean()
                half_ci = 1.96*result[:i].std(ddof=1)/sqrt(i)
                print "         %6i  % 5.1f +/- % 5.1f " % ( i, mean, half_ci )
            i += 1
            if i >= sample_size: break


# Adaptive stratified estimator
#
# The following parameter names are from the paper that describes the algorithm:
#
# N total number of samples taken in each step (i.e. across all strata)
# p probability of being in a stratum
def stratified_estimator( refm_call, agent_call, episode_length, disc_rate, samples, \
                          sample_size, dist, threads ):

    p = dist         # get probability of being in each stratum
    I = len(dist)    # number of strata, including passive
    A = sum(ceil(p)) # active strata

    # With N below I keep the steps small to start with to force the system
    # sample a reasonable number of each to get better variabilty estimates
    # before starting to adpat more.  It would be nice to have a fully online
    # version without these steps.
    N = [0, 3*A, 6*A, 10*A, \
         20*A, 30*A, 50*A, 70*A, 100*A, 250*A, 500*A, 750*A, 1000*A, \
         1250*A, 1500*A, 1750*A, 2000*A, 2500*A, 3000*A, 3500*A, 4000*A, 5000*A]

    # trim to number of program samples, or requested sample size, which ever is smaller
    max_samples = min( len(sample_data), sample_size )
    for i in range(len(N)):
        if N[i]+A >= max_samples:
            N[i] = float(max_samples)
            N = N[:i+1]
            break
        
    print "Sample size steps:"
    print N
    
    K = len(N)                 # number of adaptive stratification stages
    Y = [[] for i in range(I)] # empty collection of samples divided up by stratum
    Y[0] = [0]
    s = ones((K,I))            # estimated standard deviations for each stage & strata
    n = zeros((K,I))           # for each step the size of each stratum
    est = zeros((K))           # estimated confidence intervals

    for k in range( 1, K ):
        print

        # compute the allocations with "method a" from the paper,
        # deducting 2A from the target which is added afterward to ensure
        # all strata get at least 2 samples
        # We also divide by 2, than multiple by 2 later on to ensure that
        # the sample size in each stratum is even (this is for antithetic variables)
        
        m = (p*s[k-1])/sum(p*s[k-1]) * (N[k]-N[k-1]-2.0*A) / 2.0

        x = zeros((len(m)))
        x[0] = floor(m[0])
        for i in range(1,I):
            x[i] = max( floor(sum(m[:i+1])+0.001) - floor(sum(m[:i])), 0 )

        x *= 2.0

        # make sure each non-zero probability stratum gets sampled at least twice
        M = x + 2.0*ceil(p)
        
        # create parallel computation pool
        if threads == 0:
            pool = Pool() # default threads = core count
        else:
            pool = Pool(threads)
        results = []

        # add samples to processing pool (we skip stratum 0 which is passive)
        for i in range(1,I):
            for j in range(int(M[i])/2): # /2 is due to sampling each program twice

                if len(samples[i]) == 0:
                    print "Error: Run out of program samples in stratum: " + str(i)
                    sys.exit()

                program = samples[i].pop(0)
                args = (refm_call, agent_call, episode_length, disc_rate, i, program)
                result = pool.apply_async( test_agent, args )
                results.append( result )

        # collect the results, adding new jobs to the pool for any failed runs
        while results != []:
            result = results.pop(0)

            if not result.ready():
                # put back in the results list at the end and sleep for a moment
                results.append( result )
                sleep(0.02)
            else:
                # completed, now get the results
                stratum, perf1, perf2 = result.get(100)
                
                if isnan(perf1) or isnan(perf2):
                    # run failed so get a new sample and add to processing pool
                    #print "Adding extra sample to the pool due to run failure"
                    if len(samples[stratum]) == 0:
                        print "Error: Run out of program samples in stratum: " \
                              + str(stratum)
                        sys.exit()

                    program = samples[stratum].pop(0)
                    args = (refm_call, agent_call, episode_length, disc_rate, stratum, program)
                    result = pool.apply_async( test_agent, args )
                    results.append( result )
                else:
                    # run succeeded, so add the result to our results table Y
                    Y[stratum].append( (perf1, perf2) )

        pool.close()
        pool.join()


        # compute new total program sample counts for each strata
        n[k] = n[k-1] + M

        # compute empirical standard deviations for each stratum
        for i in range(1,I):
            if p[i] > 0.0 and n[k][i] > 2:

                YA = array(Y[i])
                sample1 = YA[:,0] # positive antithetic runs
                sample2 = YA[:,1] # negative antithetic runs

                s1 = sample1.std(ddof=1) # 1 degree of freedom
                s2 = sample2.std(ddof=1) # 1 degree of freedom
                covariance = cov( sample1, sample2 )[0,1] # default is 1 df

                var = 0.25 * ( s1*s1 + s2*s2 + 2.0 * covariance )
                s[k,i] = sqrt( var )
            else:
                s[k,i] = 1.0

        # report current estimates by strata
        for i in range(1,I):
            print " % 3d % 4d % 5d" % (i, int(M[i]), n[k][i] ),

            if n[k][i] == 0: 
                # no samples, so skip mean and half CI
                print
            elif n[k][i] < 4: 
                # don't report half CI with less than 4 program samples
                print " % 6.1f" % (array(Y[i]).mean() )
            else:
                # do a full report
                # statistical samples is twice program samples due to antithetic vars
                print " % 6.1f +/- % 5.1f" \
                   % (array(Y[i]).mean(), 1.96*s[k,i]/sqrt(n[k][i]) )
                    
        # compute the current estimate and 95% confidence interval
        for i in range(1,I):
            if p[i] > 0.0:
                est[k-1] += p[i]/( n[k][i]) * array(Y[i]).sum()

        delta = 1.96 * sum( p*s[k] ) / sqrt( N[k] )
        
        # wait until after 3rd stage due to unreliable early statistics
        if k >= min(3,K-1):
            print "\n         %6i   % 5.1f +/- % 5.1f " % (N[k], est[k-1], delta )
        
    return



# load the pre-sampled programs
sample_data = None

def load_samples( refm, cluster_node, simple_mc ):

    global sample_data

    program_sample_filename = "./refmachines/samples/" + str(refm) \
                              + cluster_node + ".samples"

    print "Loading program samples: " + program_sample_filename

    file = open( program_sample_filename )

    num_strata = 0
    sample_data = []
    for line in file:
        s, prog = line.split()
        stratum = int(s)
        num_strata = max( num_strata, stratum )
        sample_data.append( ( stratum, prog ) )
    file.close()

    num_strata += 1 # due to strata starting at 0
    num_samples = len( sample_data )

    samples = [[] for i in range(num_strata)]
    dist = zeros( (num_strata) )

    for stratum, program in sample_data:
        samples[ stratum ].append( program )
        dist[ stratum ] += 1.0/num_samples

    print "Number of program samples:" + str(num_samples)
    if not simple_mc:
        print "Number of strata:        " + str(num_strata)
        print "Strata distribution:"
        print str(dist[1:])
    print

    return samples, dist


# print basic usage
def usage():
    print "python AIQ -r reference_machine[,param1[,param2[...]]] " \
        + "-a agent[,param1[,agent_param2[...]]] " \
        + "-d discount_rate [-s sample_size] [-l episode_length] " \
        + "[-n cluster_node] [-t threads] [--log] [--simple_mc]" \


# main function that just sets things up and then calls the sampler
logging  = False
log_file = None

def main():

    global logging, log_file

    print
    print "AIQ version 1.0"
    print

    # get the command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "r:d:l:a:n:s:t:",
                                   ["help", "log","simple_mc"])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    agent_str      = None
    refm_str       = None
    disc_rate      = None
    episode_length = None
    cluster_node   = ""
    simple_mc      = False
    sample_size    = None
    agent_params   = []
    refm_params    = []
    threads        = 0

    # exit on no arguments
    if opts == []:
        usage()
        sys.exit()

    # parse arguments
    for opt, arg in opts:
        if opt == "-a":
            args = arg.split(",")
            agent_str = args.pop(0)
            for a in args:
                agent_params.append( float(a) )
            
        elif opt == "-r": 
            args = arg.split(",")
            refm_str = args.pop(0)
            for a in args:
                refm_params.append( float(a) )
        
        elif opt == "-d": disc_rate          = float(arg)
        elif opt == "-l": episode_length     = int(arg)
        elif opt == "-s": sample_size        = int(arg)
        elif opt == "-n": cluster_node       = "_"+arg
        elif opt == "-t": threads            = int(arg)
        elif opt == "--log":       logging   = True
        elif opt == "--simple_mc": simple_mc = True
        else:
            print "Unrecognised option"
            usage()
            sys.exit()

    # basic parameter checks
    if agent_str      == None: raise NameError("missing agent")
    if refm_str       == None: raise NameError("missing reference machine")
    if disc_rate      == None: disc_rate = 1.0
    if logging and simple_mc:  raise NameError("Simple mc doesn't do logging")
    if agent_str      == "Manual" and not simple_mc:
        raise NameError("Manual control only works with the simple mc sampler")

    # compute episode_length to have 95% of the infinite total in each episode
    # or if episode_length given compute the proportion that this gives
    proportion_of_total = 0.95
    if episode_length == None:
        if disc_rate == 1.0:
            print "With a discount rate of 1.0 you must set the episode length."
            print
            usage()
            sys.exit()
        else:
            episode_length = int(log(1.0-proportion_of_total)/log(disc_rate))
    else:
        proportion_of_total = 1.0-disc_rate**episode_length

    # construct refrence machine
    refm_call = refm_str + "." + refm_str + "( "
    if len(refm_params) > 0: refm_call += str(refm_params.pop(0))
    for param in refm_params: refm_call += ", " + str(param)
    refm_call += " )"
    refm = eval( refm_call )

    # construct agent
    agent_call = agent_str + "." + agent_str + "( refm, " + str(disc_rate )
    for param in agent_params: agent_call += ", " + str(param)
    agent_call += " )"
    agent = eval( agent_call )

    # report settings
    print "Reference machine:       " + str(refm)
    print "RL Agent:                " + str(agent)
    print "Discount rate:           " + str(disc_rate)
    print "Episode length:          " + str(episode_length),
    if disc_rate != 1.0:
        print " which covers %3.1f%% of the infinite geometric total" \
              % (100.0*proportion_of_total)
    else:
        print

    if agent == "Manual()" and not simple_mc:
        print "Error: Manaual agent only works with simple_mc sampling"
        sys.exit()

    if disc_rate != 1.0 and proportion_of_total < 0.75:
        print
        print "WARNING: The episode length is too short for this discout rate!"
        print

    print "Sample size:             " + str(sample_size)

    # load in program samples
    samples, dist = load_samples( refm, cluster_node, simple_mc )

    if sample_size == None:
        sample_size = len(sample_data)
        
    # The following is a crude check as we can still run out of samples in a
    # stratum depending on how the adaptive stratification decides to sample.
    if sample_size > 2.0 * len(sample_data):
        print
        print "Error: More samples have been requested than are available in " \
              "the program sample file! (including fact that they are sampled twice)"
        sys.exit()

    # report logging
    if logging:
        log_file_name = "./log/" + str(refm) + "_" + str(disc_rate) + "_" \
                        + str(episode_length) + "_" + str(agent) + cluster_node \
                        + strftime("_%Y_%m%d_%H_%M_%S",localtime()) + ".log" 
        log_file = open( log_file_name, 'w' )
        for i in range( 1, len(dist) ):
            log_file.write( str(dist[i]) + " " )
        log_file.write("\n")
        log_file.flush()
        print "Logging to file:         " + log_file_name

    # run an estimation algorithm
    if simple_mc:
        simple_mc_estimator( refm_call, agent_call, episode_length, disc_rate, sample_size )
    else:
        # Kill agent and pass in its constructor call, this is because on Windows
        # some agents have trouble serialising which messes up the multiprocessing
        # library that Python uses.  Easier just to construct the agent inside the
        # method that gets called in parallel.
        agent = None 
        stratified_estimator( refm_call, agent_call, episode_length, disc_rate,
                              samples, sample_size, dist, threads )

    # close log file
    if logging: log_file.close()

    
if __name__ == "__main__":
    main()




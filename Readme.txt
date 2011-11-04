

AIQ README
==========


For the theory behind this see:

Algorithm Intelligence Quotient: A practial measure of machine intelligence
by Shane Legg and Joel Veness

and for the theory behind that see:

Universal Intelligence: A formal definition of machine intelligence
by Shane Legg and Marcus Hutter

The code is released under the GNU GPLv3.  See Licence.txt file.



Outline of files and directories:
---------------------------------


AIQ.py 

This is the main program to use to compute an agent's AIQ value for a
given reference machine, discount rate etc.

Arguments:

-a agent_name,param1,param2,...   

-r ref_machine_name,param1,param2,...

-d discount_rate

-l episode_length (if you don't give it it will compute one for you
 with 95% of the total infinite length episode reward covered)

-s sample_size Size of sample to use.

-n cluster_node Name of the cluster node.  Used for naming output log
 files and which sample file to read (need to think about latter
 aspect, maybe change).

-t threads_to_use Default is the number of cores on the machine.  But
 for a multiple threaded agent you might want to set it to be less

--log  Swtich on output logging

--simple_mc Use a simple MC sample rather than the stratified sampler.
  Useful for sanity checks and also debugging as it doesn't do any
  async stuff etc.


An example run of AIQ would be:

python AIQ.py -r BF,3 -d 0.999 -a Q_l,50.0,0.5,0.5,0.05

which is a BF reference machine with a 3 symbol tape, discounting of
0.999 (which implies an episode length of 2994 -- it tells you this)
and Q lambda with parameters ....

If you want to try BF with, say, a 14 symbol tape, you'll first need
to generate a program sample file for this (see below).



ComputeFromLog.py 

Give it a log file name and it will compute the AIQ as well as results
for each strata.  At the moment you can't combined logs simply because
the first line is the program sample distribution information (needed
to work out the stratified estimate of the AIQ value).

Maybe at some point I'll let the program accept multiple log files as
input.  You could simply strip the first line of all logs except the
first one and then concatenate them for now.



/log

The raw results of runs are dumped here if logging is turned on with
the -log option in AIQ.

First line of a log file is the estimated strata distribution computed
by AIQ from the sample file, followed by lines containing a time
stamp, strata number, and result.  The file name contains the Agent,
Reference machine, discount, episode length.  You can then quickly
compute the AIQ from one of these log files later on using the program
ComputeFromLog.py (see above)



/agents

This contains the code for the various agents.

Agent.py  Base class for agents

Random.py Agent that takes random actions

Freq.py  Slightly smarter agent that looks at reward associated with actions.

Q_l.py  Q learning with eligibility traces.

HLQ_l.py  Like Q learning but with an automatic learning rate.

MC_AIXI.py Wrapper for Monte Carlo AIXI agent.  Must have an
executable call mc-aixi in this directory in order to run.  C++ code
for MC AIXI can be downloaded from the internet.



/refmachines

This contains the code for the reference machines.

ReferenceMachine.py  Base class for reference machines

BF.py BF based reference machine.  Take parameters for the number of
symbols (i.e. alphabet size, default is 3) and the number of cells
that the observations use (default is 1).  Actions and rewards are
still fixed at 1 tape cell.

BF_sampler.py Generates samples of BF programs, works out their
strata, and outputs these to the terminal.  You'll want to stick these
in a sample file.  You have to name the file correctly yourself to
match what AIQ expects.  The -s option tell it how many samples to
generate. The file consists of just rows of samples so you can
concatenate the output of different runs to make a combined sample
file.


/refmachine/sample 

Directory of program samples along with there strata.  Files
are named by the reference machine (including parameters) followed by
.sample.  This saves AIQ having to generate new samples and work out
what strata they are in, and AIQ also computes the estimated true
strata probabilities based on this sample.  So make it reasonable
large.  Say 100k programs for proper tests.  Use BF_sampler.py to
generate these for the BF reference machine.


#
# Estimate AIQ from a file of log results
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3


from scipy import ones, zeros, floor, array, sqrt, cov

import getopt, sys

from os.path import basename



def estimate( file, detailed ):

    # load in the strata distribution
    dist_line = ["0.0"]
    dist_line += file.readline().split()
    dist = array( dist_line, float )

    p = dist      # probabilyt of a program being in each strata
    I = len(dist) # number of strata, including passive
    A = I-1       # active strata


    Y = [[] for i in range(I)] # empty collection of samples divided up by stratum
    Y[0] = [0]
    s = ones((I))            # estimated standard deviations for each stage & strata

    # read in log file results
    num_samples = 0
    for result in file:
        stamp, stratum, perf1, perf2 = result.split()
        z = int(stratum)
        if True: #z > 10:
            Y[int(stratum)].append( (float(perf1),float(perf2)) )
            num_samples += 2

    # compute empirical standard deviations for each stratum
    for i in range(1,I):
        if p[i] > 0.0 and len(Y[i]) > 2:

            YA = array(Y[i])
            sample1 = YA[:,0] # positive antithetic runs
            sample2 = YA[:,1] # negative antithetic runs

            s1 = sample1.std(ddof=1) # 1 degree of freedom
            s2 = sample2.std(ddof=1) # 1 degree of freedom
            covariance = cov( sample1, sample2 )[0,1] # default is 1 df

            var = 0.25 * ( s1*s1 + s2*s2 + 2.0 * covariance )
            s[i] = sqrt( var )
        else:
            s[i] = 1.0


    # report current estimates by strata
    if detailed:
        for i in range(1,I):
            stratum_samples = len(Y[i])*2.0
            print " % 3d % 5d" % (i, stratum_samples ),

            if stratum_samples == 0:
                # no samples, so skip mean and half CI
                print
            elif stratum_samples < 4:
                # don't report half CI with less than 4 samples
                print " % 6.1f" % (array(Y[i]).mean() )
            else:
                # do a full report
                print " % 6.1f +/- % 5.1f" \
                   % (array(Y[i]).mean(), 1.96*s[i]/sqrt(stratum_samples) )

        print

    # compute the current estimate and 95% confidence interval
    est = 0.0
    for i in range(1,I):
        stratum_samples = len(Y[i])*2.0
        if p[i] > 0.0 and stratum_samples > 2:
            est += p[i]/stratum_samples * array(Y[i]).sum()

    delta = 1.96 * sum(p*s) / sqrt(num_samples)

    print "%6i  % 5.1f +/- % 5.1f" % (num_samples, est, delta ),

    return


# print basic usage
def usage():
    print "python ComputeFromLog [--full] log_file_name [log_file_name ...]" 


# main function that just sets things up and then calls the sampler
logging  = False
log_file = None

def main():

    global logging, log_file

    detailed = False

    print
    print "Compute AIQ from log file results, version 1.0"
    print

    sys.argv.pop(0)

    if len(sys.argv) == 0:
        usage()
        sys.exit()

    if sys.argv[0] == "--full":
        detailed = True
        sys.argv.pop(0)

    if len(sys.argv) == 0:
        usage()
        sys.exit()

    for file_name in sys.argv:
        file = open( file_name, 'r')
        estimate( file, detailed )
        print ":" + basename(file_name)
        if detailed: print
        file.close()
    

    
if __name__ == "__main__":
    main()


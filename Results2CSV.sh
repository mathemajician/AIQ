#!/bin/sh
# BASH script to convert ComuteFromLog output to a CSV format
tail -n +4 $1 | sed s/\ +\\/-\ /,/ | sed s/\ :/,\ / | sed s/\ \ log\\//,\ / | \
                sed s/_/\ / | sed s/_/,\ / | sed s/_/,\ / | \
                tr '(' ',' | tr ')' ',' | sed s/,_.*// | cut -c9-

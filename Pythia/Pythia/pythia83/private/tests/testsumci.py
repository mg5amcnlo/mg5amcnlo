# Script to read testsums.log, and provide exit codes readable by 
# the gitlab CI framework.
from sys import exit
import argparse
parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument("-d", "--diff", dest="DIFF",default=False,
        action="store_true", help="Test if 'Differed' is zero.")
parser.add_argument("-f", "--fail", dest="FAIL",default=False,
        action="store_true", help="Test if 'Failed' is zero.")

args = parser.parse_args()
ndiff = 0
nfail = 0
with open("../../testsums.log") as f:
    for line in f:
        l = line.split()
        try:
            if l[0] == "Differed:":
                ndiff += int(l[1])
            if l[0] == "Failed:":
                nfail += int(l[1])
        except:
            continue
if args.DIFF:
    if ndiff > 0:
        exit(1)
    exit(0)

if args.FAIL:
    if nfail > 0:
        exit(1)
    exit(0)

if nfail+ndiff > 0:
    exit(1)
exit(0)


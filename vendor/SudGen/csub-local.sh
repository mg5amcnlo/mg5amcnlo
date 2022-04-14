#!/bin/bash
# usage:
#    condor_sub queue nmasses command cmd-arguments
# Important note:
#  * output needs to be specified in command line, nothing will be
#    copied over at the end of the run!
#  * in all command line arguments, "SEED"
#    is replaced by the corresponding mass
#
# 
# queue options:
#   * espresso     = 20 minutes
#   * microcentury = 1 hour
#   * longlunch    = 2 hours
#   * workday      = 8 hours
#   * tomorrow     = 1 day
#   * testmatch    = 3 days
#   * nextweek     = 1 week
#

fn="csub-lxplus.sh"
fnmass="masses-condor.txt"
fnsubmit="submit-condor.sh"

function general_help {
    echo "condor_sub.sh is a script for submitting jobs with HTCondor
usage:
   condor_sub queue nmasses command cmd-arguments
Important note:
 * In fnoutput and all command line arguments, SEED
   is replaced by the corresponding mass numbers.
 * Output files need to be specified in command line, nothing will be
   copied at the end of the run.
"
}

function queue_help {
    echo "Queue options:"
    echo "  * espresso     = 20 min"
    echo "  * microcentury = 1 hour"
    echo "  * longlunch    = 2 hours"
    echo "  * workday      = 8 hours"
    echo "  * tomorrow     = 1 day"
    echo "  * testmatch    = 3 days"
    echo "  * nextweek     = 1 week
"
}

function validate_queue {
    if [ "$1" != "espresso" ] && [ "$1" != "microcentury" ] \
	   && [ "$1" != "longlunch" ] && [ "$1" != "workday" ] \
	   && [ "$1" != "tomorrow" ] && [ "$1" != "testmatch" ] \
	   && [ "$1" != "nextweek" ]
    then
	queue_help
	echo "Error: "$1" is not valid"
	exit
    fi
}

function write_masses {
    nmass=$((10#$1))
    if ! [ -f masses-condor.txt ]
    then
       for ((i=1; i<=nmass; i++)); do
	   se=$(printf "%0.${#1}d" $i)
	   echo $se >> $fnmass
       done
    else
	echo
	echo "WARNING: file "$fnmass" already exists, will use this."
	echo
    fi
     
}

function write_submit {
    if ! [ -f $fnsubmit ] ; then
    echo "#!/bin/bash
ARGUMENTS='\$(argument)'
EXECUTABLE=$2
LOGDIR=logs
if ! [ -d \$LOGDIR ]
then
  mkdir -p \$LOGDIR
fi
echo '
executable = '\$EXECUTABLE'
output = '\$LOGDIR'/\$(argument).stdout
error = '\$LOGDIR'/\$(argument).stderr 
log = '\$LOGDIR'/\$(argument).log 
arguments = \"${@:3}\"
getenv = True
periodic_release =  (NumJobStarts < 10) && ((CurrentTime - EnteredCurrentStatus) > 30)
+JobFlavour = \"$1\"
queue argument from $fnmass' | condor_submit" | sed 's/MASS/$(argument)/g' > $fnsubmit
    chmod +x $fnsubmit
    else
	echo
	echo "WARNING: "$fnsubmit" already exists, will use this."
	echo
    fi
}

if (($# > 2)) ; then
    validate_queue $1
    write_masses $2
    cmdline=${@:3}
    cmdlineargs=${@:4}
    exec=$(echo $3 | sed 's/.\///' )
    write_submit $1 $exec $cmdlineargs
    echo "Creating files "$fnmass", "$fnsetup" and "$fnsubmit"."
    echo "  * start jobs with './"$fnsubmit"'"
    echo "  * use '"$fn" clean' to delete submission files"
elif [ "$1" == "clean" ] ; then
    echo "Removing "$fnmass", "$fnsetup" and "$fnsubmit
    rm $fnmass $fnsetup $fnsubmit
    exit
else
    general_help
    queue_help
    echo "Arguments required: nmasses queue command"
    exit
fi

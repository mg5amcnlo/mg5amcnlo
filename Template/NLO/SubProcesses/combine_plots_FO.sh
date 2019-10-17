#!/bin/bash

# $? is the value of last executed command. A call to this function
# after a failure will cause the program to quit the script
function teststatus {
rc=$?
if [ 0 = $rc ]
then
:
else
echo $*, exit status=$rc 
exit $rc
fi
}

function combine {
counter=0
i=1
echo -n "" > dir
for plot_file in "${@:2}" ; do
    counter=$(($counter + 1))
    echo $plot_file >> dir
    if [[ $(($counter % 40)) == 0 ]]; then
        echo -n "" > dir2
	echo "40" > dir2
	cat dir >> dir2
	./read40 < dir2
        teststatus Failure in step 1
	rm -f dir
	rm -f dir2
	mv fort.88 $1_$i\.top 
	mv read40.out "S1read40_"$1_$i\.out
	i=$(($i + 1))
	echo -n "" > dir
    fi
done
if [[ $(($counter % 40)) != 0 ]] ; then
    echo $(($counter % 40)) > dir2
    cat dir >> dir2
    ./read40 < dir2
    teststatus Failure in step 2
    rm -f dir
    rm -f dir2
    mv fort.88 $1_$i\.top 
    mv read40.out "S2read40_"$1_$i\.out
fi
}

make read40
rm -f MADatNLO*.top

if [ "$#" -gt 64000 ]; then
    combine "MADatNLO_A" "$@"
    combine "MADatNLO_B" MADatNLO_A*.top
    combine "MADatNLO_C" MADatNLO_B*.top
    combine "MADatNLO_D" MADatNLO_C*.top
    mv MADatNLO_D_1.top MADatNLO.top
elif [ "$#" -gt 1600 ]; then
    combine "MADatNLO_A" "$@"
    combine "MADatNLO_B" MADatNLO_A*.top
    combine "MADatNLO_C" MADatNLO_B*.top
    mv MADatNLO_C_1.top MADatNLO.top
elif [ "$#" -gt 40 ]; then
    combine "MADatNLO_A" "$@"
    combine "MADatNLO_B" MADatNLO_A*.top
    mv MADatNLO_B_1.top MADatNLO.top
elif [ "$#" -gt 1 ]; then
    combine "MADatNLO_A" "$@"
    mv MADatNLO_A_1.top MADatNLO.top
else
    cp $1 MADatNLO.top
fi


if [ -f read40.errors ]
then
rm -f read40.errors
fi

find . -name S"*"read40"*".out -exec fgrep -il "READ40 ERROR READ40 ERROR" {} \; > read40.errors

if [ ! -s read40.errors ] ; then
    echo 'no errors found'
else
    echo 'ERRORS! see read40.errors for details'
fi

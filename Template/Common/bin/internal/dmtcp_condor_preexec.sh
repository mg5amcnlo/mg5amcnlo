DAGMANPATH=$1
DAGMANFLOW=$2

DAGMANJOBID="$(condor_q -nobatch -autoformat ClusterId \
  -const "regexp(\"${DAGMANFLOW}\", Arguments) && regexp(\"condor_dagman\", Cmd)" | head -n1)"

mkdir -p $DAGMANPATH/dmtcp_$DAGMANJOBID

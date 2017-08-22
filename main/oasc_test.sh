#! /bin/bash
#
# Launch with:
# sh xxx.sh autok tasks.txt
#
#
currentdir="`pwd`"


CMD='python '$currentdir'/../oasc/result.py '
JOBS=$currentdir'/'$2
# JOBS=$currentdir'/tasks-k.txt'
LOCK=$currentdir'/run_test.lock'
COMPLETED=$currentdir'/completed_folds.log'
ERRDIR=$currentdir'/running'
HOSTNAME="`hostname`"
ERR=$ERRDIR'/task_'$$'_'$HOSTNAME

mkdir -p $ERRDIR

while
  [ -s $JOBS  ]
do

  lockfile $LOCK

  job=`head -n 1 $JOBS`
  if
    [ -z "$job" ]
  then
    rm -f $LOCK
    continue
  fi
  cat $JOBS | sed '1d' > $ERR 
  cat $ERR > $JOBS # sliced content management
  echo "$job|$1" > $ERR  # save running task
  rm -f $LOCK

  scenario=`echo $job | awk -F\| '{print $1}'`


  echo 'Testing Scenario: '$scenario

  $CMD "$scenario" "$1"

  lockfile $LOCK
  echo $job >> $COMPLETED 

  rm -f $ERR
  rm -f $LOCK
  
  echo 'completed'

done
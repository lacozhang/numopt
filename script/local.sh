#!/bin/bash
# set -x
dir=`dirname "$0"`
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi

num_servers=$1
echo "servers: $num_servers"
shift
num_workers=$1
echo "workers: $num_workers"
shift
bin=$1
echo "bin path: $bin"
shift
arg="-num_servers ${num_servers} -num_workers ${num_workers} -log_dir log $@" #" -app ${dir}/$@"
echo "args : $arg"


# killall -q $(basename ${bin})
# killall -q ${bin}
HEAPCHECK=draconian
# start the scheduler
Sch="role:SCHEDULER,hostname:'127.0.0.1',port:8001,id:'H'"
echo "scheduler node : $Sch"
${bin} -my_node ${Sch} -scheduler ${Sch} ${arg} &

# start servers
for ((i=0; i<${num_servers}; ++i)); do
    port=$((9600 + ${i}))
    N="role:SERVER,hostname:'127.0.0.1',port:${port},id:'S${i}'"
    # HEAPPROFILE=/tmp/S${i} \
    # CPUPROFILE=/tmp/S${i} \
    echo "node info $N"
    ${bin} -my_node ${N} -scheduler ${Sch} ${arg} &
done

# start workers
for ((i=0; i<${num_workers}; ++i)); do
    port=$((9500 + ${i}))
    N="role:WORKER,hostname:'127.0.0.1',port:${port},id:'W${i}'"
    # HEAPPROFILE=/tmp/W${i} \
    # CPUPROFILE=/tmp/W${i} \
    echo "node info $N"
    ${bin} -my_node ${N} -scheduler ${Sch} ${arg} &
done

wait

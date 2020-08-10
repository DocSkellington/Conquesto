#!/usr/bin/env bash
# Copyright © - 2020 - UMONS
# CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
# You should have received a copy of the BSD-3 Clause License along with this program. 
# Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).

# This script runs clingo (using Generate and Test programs with cardinality constraints) and writes some statistics in a file

function output_bool {
    if [[ $1 = true ]]
    then
        echo -n "True " >> ${output_file}
    else
        echo -n "False " >> ${output_file}
    fi
}

function run_clingo {
    reached_timelimit=false
    { time timeout $1s clingo --stats=1 $2 $3 > ${tmp_path}/out.txt 2>& 1 ; } 2> ${tmp_path}/time.txt
    success=$?
    # We check if the command was timeout, or not
    # 124 means timeout
    if [ ${success} -eq 124 ]
    then
        reached_timelimit=true
        echo "Timeout"
    fi
    # We parse the output
    cpu_time=0
    choices=0
    rules=0
    atoms=0
    bodies=0
    variables=0
    constraints=0
    reached_timelimit=true
    satisfiable=false
    # First, clingo's output
    while read line;
    do
        # Does the line contain the word "Choices"?
        if [[ ${line} =~ "Choices" ]]
        then
            choices=$(echo ${line} | cut -d' ' -f3)
        elif [[ ${line} =~ "Rules" ]]
        then
            rules=$(echo ${line} | cut -d' ' -f3)
        elif [[ ${line} =~ "Atoms" ]]
        then
            atoms=$(echo ${line} | cut -d' ' -f3)
        elif [[ ${line} =~ "Bodies" ]]
        then
            bodies=$(echo ${line} | cut -d' ' -f3)
        elif [[ ${line} =~ "Variables" ]]
        then
            variables=$(echo ${line} | cut -d' ' -f3)
        elif [[ ${line} =~ "Constraints" ]]
        then
            constraints=$(echo ${line} | cut -d' ' -f3)
        elif [[ ${line} =~ "UNSATISFIABLE" ]]
        then
            satisfiable=false
            reached_timelimit=false
        elif [[ ${line} =~ "SATISFIABLE" ]]
        then
            satisfiable=true
            reached_timelimit=false
        fi
    done < ${tmp_path}/out.txt

    # Then, time's output
    while read line;
    do
        # User is the CPU time
        if [[ ${line} =~ "user" ]]
        then
            # We extract only the seconds
            cpu_time=$(echo ${line} | cut -d' ' -f2 | cut -d'm' -f2 | cut -d's' -f1)
        fi
    done < ${tmp_path}/time.txt

    output_bool ${reached_timelimit}
    output_bool ${satisfiable}
    echo -n "${cpu_time} ${choices} ${rules} ${atoms} ${bodies} ${variables} ${constraints} " >> ${output_file}
}

program=clingo
database_type=$1
min_n=$2
max_n=$3
timeout=${4:-100} # optional argument
echo ${program} ${database_type} ${min_n} ${max_n} ${timeout}

if [ -z "${database_type}" ]
then
    echo "The second argument must be the database type. It must be \"yes\", \"no\", or \"random\""
    exit 1
fi
if [ "${database_type}" != 'yes' ] && [ "${database_type}" != 'no' ] && [ "${database_type}" != 'random' ]
then
    echo "${database_type} is not a valid database type. It must be \"yes\", \"no\", or \"random\""
    exit 1
fi

if [ -z "${min_n}" ]
then
    echo "The third argument must be the minimal alpha value."
    exit 1
fi
if [ -z "${max_n}" ]
then
    echo "The third argument must be the maximal alpha value."
    exit 1
fi

tmp_path="tmp-${database_type}-${program}-cardinality"
if [ ! -d ${tmp_path} ]
then
    mkdir ${tmp_path}
fi

n_queries=$(python3 conquesto/generate_files.py 2> /dev/null)
time_limit=100

db_path=""
if [ "${database_type}" = "yes" ]
then
    db_path="${tmp_path}/yes.lp"
elif [ "${database_type}" = "no" ]
then
    db_path="${tmp_path}/no.lp"
elif [ "${database_type}" = "random" ]
then
    db_path="${tmp_path}/random.lp"
fi

output_file="generated/dataset_varying_${database_type}_${program}_cardinality_${min_n}_${max_n}_${n_queries}_${time_limit}.txt"
# If the file already exists, we delete it (in order to start a clean dataset)
if [ -e ${output_file} ]
then
    rm ${output_file}
fi

# The description of the dataset
echo "${database_type} ${time_limit} ${n_queries} 2 3 2" >> ${output_file}

for (( query=0 ; query<${n_queries} ; query=$query+1))
do
    echo "$query / ${n_queries}"

    # We generate the programs
    python3 conquesto/generate_files.py ${database_type} ${query}

    for (( n=${min_n} ; n<=${max_n} ; n=$n+1))
    do
        echo ${min_n} ${n} ${max_n}

        # We generate the databases
        db_size=$(python3 conquesto/generate_files.py ${database_type} ${program} ${query} ${n} "cardinality")

        echo -n "${n} ${db_size} ${query} " >> ${output_file}

        # We run both GC and FO
        # Note that the order is important in order for generate_figures.py to work
        for algorihtm in gc fo
        do
            program_file=${tmp_path}/${algorihtm}.lp
            run_clingo ${time_limit} ${program_file} ${db_path}

            # Some "asserts"
            if [ ${reached_timelimit} = false ]
            then 
                if [ "${database_type}" = "yes" ]
                then
                    if [ "${algorihtm}" = gc ] && [ ${satisfiable} = true ]
                    then
                        echo "The output of gc is incorrect. It is true but should be false"
                        exit 2
                    elif [ "${algorihtm}" = fo ] && [ ${satisfiable} = false ]
                    then
                        echo "The output of fo is incorrect. It is false but should be true"
                        exit 2
                    fi
                elif [ "${database_type}" = "no" ]
                then
                    if [ "${algorihtm}" = gc ] && [ ${satisfiable} = false ]
                    then
                        echo "The output of gc is incorrect. It is false but should be true"
                        exit 2
                    elif [ "${algorihtm}" = fo ] && [ ${satisfiable} = true ]
                    then
                        echo "The output of fo is incorrect. It is true but should be false"
                        exit 2
                    fi
                fi
            fi
        done
        echo "" >> ${output_file}
    done
done

rm -r ${tmp_path}

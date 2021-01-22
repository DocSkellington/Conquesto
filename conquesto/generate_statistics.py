""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
from typing import List, Tuple, Dict, Any
import os
import tempfile
import subprocess
from collections import namedtuple
import pickle
import enum
import time

import numpy as np

import query.query as query
from generators.random_generator import RandomGenerator
from generators.exhaustive_generators import ExhaustiveGenerator, FixedSchemaExhaustiveGenerator

class DatabaseType(enum.Enum):
    YES = "yes"
    NO = "no"
    RANDOM = "random"

class Program(enum.Enum):
    CLINGO = "clingo"
    DLV = "DLV"

class Results:

    def __init__(
        self,
        timeout: bool,
        cpu_time : float,
        satisfiable : bool = False,
        choices : int = -1,
        rules : int = -1,
        atoms : int = -1,
        bodies : int = -1,
        variables : int = -1,
        constraints : int = -1
    ):
        self.timeout = timeout
        self.cpu_time = cpu_time
        self.satisfiable = satisfiable
        self.choices = choices
        self.rules = rules
        self.atoms = atoms
        self.bodies = bodies
        self.variables = variables
        self.constraints = constraints

    def __str__(self) -> str:
        return "Timeout: {}\nSatisfiable: {}\nCPU Time: {}\nChoices: {}\nRules: {}\nAtoms: {}\nBodies: {}\nVariables: {}\nConstraints: {}\n".format(self.timeout, self.satisfiable, self.cpu_time, self.choices, self.rules, self.atoms, self.bodies, self.variables, self.constraints)

    def to_list(self) -> List[str]:
        return [str(self.timeout),\
            str(self.satisfiable),
            str(self.cpu_time),\
            str(self.choices),\
            str(self.rules),\
            str(self.atoms),\
            str(self.bodies),\
            str(self.variables),\
            str(self.constraints)]

# Database type, time limit, table number, number of positions for each table
Description = namedtuple("Description", ["databaseType", "timeLimit", "numberOfQueries", "numberOfTables", "positionsPerTable"])
# Parameter of the size of the database, the actual size of the db, GC results, FO results
Line = namedtuple("Line", ["sizeParameter", "actualSize", "queryID", "gc", "fo"])
Dataset = List[Line]

def launch_program(
    f_code : str,
    f_db : str,
    program : Program,
    time_limit : int = 60,
    parallel_mode : int = 0
) -> Results:
    '''
    Launches a program and gets the time taken to execute it

    :param f_code: The file where the code of the program is contained
    :param f_db: The file where the db code is contained
    :param program: The program to use
    :param time_limit: The maximal time allowed
    :param parallel_mode: The number of CPU cores to use (only for clingo; 0 disables parallel mode)
    :return: The CPU time taken by the program to stop
    '''
    timeout= False
    values : Dict[str, Any] = {}
    if program == Program.CLINGO:
        with tempfile.TemporaryFile("w+") as result_file:
            if parallel_mode == 0:
                subprocess.run(["clingo", f_code, f_db, "--stats=1", "--time-limit={}".format(time_limit)], stdout=result_file)
            else:
                subprocess.run(["clingo", f_code, f_db, "--stats=1", "--parallel-mode", str(parallel_mode), "--time-limit={}".format(time_limit)], stdout=result_file)

            result_file.seek(0)

            for line in result_file:
                if "CPU Time" in line:
                    values["cpu_time"] = float(''.join(c for c in line if (c.isdigit() or c == '.')))
                elif "Choices" in line:
                    values["choices"] = int(''.join(c for c in line if c.isdigit()))
                elif "Rules" in line:
                    values["rules"] = int(''.join(c for c in line if c.isdigit()))
                elif "Atoms" in line:
                    values["atoms"] = int(''.join(c for c in line if c.isdigit()))
                elif "Bodies" in line:
                    values["bodies"] = int(''.join(c for c in line if c.isdigit()))
                elif "Variables" in line:
                    values["variables"] = int(list(filter(lambda c: c.isdigit(), line.split(" ")))[0])
                elif "Constraints" in line:
                    values["constraints"] = int(list(filter(lambda c: c.isdigit(), line.split(" ")))[0])
                elif "TIME LIMIT" in line:
                    timeout = True
                elif "UNSATISFIABLE" in line:
                    values["satisfiable"] = False
                elif "SATISFIABLE" in line:
                    values["satisfiable"] = True
    elif program == Program.DLV:
        with tempfile.TemporaryFile("w+") as model_file, tempfile.TemporaryFile("w+") as stat_file:
            current_time = time.monotonic()
            try:
                # dlv writes its statistics in the error stream
                subprocess.run(["dlv", "-n=1", "-stats", f_code, f_db], stdout=model_file, stderr=stat_file, timeout=time_limit)
                values["cpu_time"] = time.monotonic() - current_time
                timeout = False
            except subprocess.TimeoutExpired:
                values["cpu_time"] = time_limit
                print("Timeout")
                timeout = True

            model_file.seek(0)
            stat_file.seek(0)

            values["satisfiable"] = False
            values["atoms"] = values["bodies"] = values["variables"] = 0 # dlv does not give these pieces of information
            values["choices"] = values["rules"] = 0 # It may happen that dlv does not give the number of choices. So, I assume it is zero
            for line in stat_file:
                if "Rules" in line:
                    values["rules"] = int(''.join(c for c in line if c.isdigit()))
                elif "Constraints" in line and "Weak" not in line:
                    values["constraints"] = int(''.join(c for c in line if c.isdigit()))
                elif "Choices" in line:
                    values["choices"] = int(''.join(c for c in line if c.isdigit()))
            for line in model_file:
                if "{" in line: # Indicates the start of a stable model
                    values["satisfiable"] = True

    if timeout or "cpu_time" not in values:
        results = Results(True, time_limit)
    else:
        print("Time taken: ", values["cpu_time"])
        results= Results(False,\
            values["cpu_time"],\
            values["satisfiable"],\
            values["choices"],\
            values["rules"],\
            values["atoms"],\
            values["bodies"],\
            values["variables"],\
            values["constraints"])

    return results

def run_varying_database_size_single_query(
    q : query.Query,
    instance_type : DatabaseType,
    program : Program,
    min_n : int,
    max_n : int,
    time_limit : int = 100,
    query_ID : int = 0
) -> Dataset:
    '''
    For the given query, runs the corresponding programs on databases of given type and of size 1 to n and returns the statistics for each run.

    :param query: The query we are testing.
    :param instance_type: Type of the database generated.
    :param max_n: Maximal value of the parameter governing the size of the databases.
    :param time_limit: Maximum time limit authorized to execute a program.
    :param query_ID: The ID of the query. It is used to distinguish the different queries, when there are multiple queries.

    :return: A list of tuples. At position i in the list, each tuple contains the results of both programs on database size i.
    '''
    data : Dataset = []
    # We create temporary files
    # We need to do this this way because we want to first write in the file with this program and be able to open it in Clingo
    gc_program_id, gc_program_path = tempfile.mkstemp(text=True)
    fo_program_id, fo_program_path = tempfile.mkstemp(text=True)
    db_id, db_path = tempfile.mkstemp(text=True)

    # We do not need the file handles
    os.close(gc_program_id)
    os.close(fo_program_id)
    os.close(db_id)

    q.write_generate_and_check(False, gc_program_path)
    q.write_fo_program(fo_program_path)

    for i in range(min_n, (max_n+1)):
        print(min_n, i, max_n)
        if instance_type == DatabaseType.YES:
            with tempfile.TemporaryFile("w") as f:
                target_size = q.write_naive_no_db_to_file(i, f)
            actual_size = q.write_naive_yes_db(i, db_path, target_size)
        elif instance_type == DatabaseType.NO:
            actual_size = q.write_naive_no_db(i, db_path)
        elif instance_type == DatabaseType.RANDOM:
            with tempfile.TemporaryFile("w") as f:
                target_size = q.write_naive_no_db_to_file(i, f)
            # We make sure that the random database is valid
            target_size = max(q.table_number, target_size)
            actual_size = q.write_random_db(target_size, db_path)
            assert target_size == actual_size, "{} != {}".format(target_size, actual_size)

        gc_results = launch_program(gc_program_path, db_path, program, time_limit=time_limit)
        fo_results = launch_program(fo_program_path, db_path, program, time_limit=time_limit)
        # We ensure that the results are coherent
        if instance_type == DatabaseType.YES:
            assert not gc_results.satisfiable
            assert fo_results.satisfiable
        elif instance_type == DatabaseType.NO:
            assert gc_results.satisfiable
            assert not fo_results.satisfiable
        data.append(Line(i, actual_size, query_ID, gc_results, fo_results))

    # We delete the temporary files
    os.remove(gc_program_path)
    os.remove(fo_program_path)
    os.remove(db_path)

    return data

def run_varying_database_size(
    queries : List[query.Query],
    instance_type: DatabaseType,
    program : Program,
    min_n : int,
    max_n : int,
    time_limit : int
) -> Dataset:
    '''
    For the given queries, runs the corresponding programs on databases of given type and of size 1 to n and returns the statistics for each run.

    :param query: The query we are testing.
    :param instance_type: Type of the database generated.
    :param max_n: Maximal value of the parameter governing the size of the databases.
    :param time_limit: Maximum time limit authorized to execute a program.

    :return: A list of tuples. At position i in the list, each tuple contains the results of both programs on database size i.
    '''
    data : Dataset = []
    for i, q in enumerate(queries):
        print((i+1), "/", len(queries), instance_type.value)
        print(q)

        subdata = run_varying_database_size_single_query(q, instance_type, program, min_n, max_n, time_limit, i)
        data += subdata

    return data

def run_fixed_database_size_single_query(
    q : query.Query,
    instance_type : DatabaseType,
    program : Program,
    n : int,
    time_limit : int = 100,
    query_ID : int = 0
) -> Dataset:
    data : Dataset = []
    # We create temporary files
    # We need to do this this way because we want to first write in the file with this program and be able to open it in Clingo
    gc_program_id, gc_program_path = tempfile.mkstemp(text=True)
    fo_program_id, fo_program_path = tempfile.mkstemp(text=True)
    db_id, db_path = tempfile.mkstemp(text=True)

    # We do not need the file handles
    os.close(gc_program_id)
    os.close(fo_program_id)
    os.close(db_id)

    q.write_generate_and_check(False, gc_program_path)
    q.write_fo_program(fo_program_path)

    if instance_type == DatabaseType.YES:
        with tempfile.TemporaryFile("w") as f:
            target_size = q.write_naive_no_db_to_file(n, f)
        actual_size = q.write_naive_yes_db(n, db_path, target_size)
    elif instance_type == DatabaseType.NO:
        actual_size = q.write_naive_no_db(n, db_path)
    elif instance_type == DatabaseType.RANDOM:
        with tempfile.TemporaryFile("w") as f:
            target_size = q.write_naive_no_db_to_file(n, f)
        actual_size = q.write_random_db(target_size, db_path)

    gc_results = launch_program(gc_program_path, db_path, program, time_limit=time_limit)
    fo_results = launch_program(fo_program_path, db_path, program, time_limit=time_limit)
    data.append(Line(n, actual_size, query_ID, gc_results, fo_results))

    # We delete the temporary files
    os.remove(gc_program_path)
    os.remove(fo_program_path)
    os.remove(db_path)

    return data

def run_fixed_database_size(
    queries : List[query.Query],
    instance_type : DatabaseType,
    program : Program,
    n : int,
    time_limit : int = 100
) -> Dataset:
    data : Dataset = []
    for i, q in enumerate(queries):
        print((i+1), "/", len(queries), instance_type.value)
        print(q)

        subdata = run_fixed_database_size_single_query(q, instance_type, program, n, time_limit, i)
        data += subdata

    return data

def save_results(
    description : Description,
    dataset : Dataset,
    filepath : str
):
    '''
    Saves the description and the dataset in a file

    :param description: The description of the dataset
    :param dataset: The dataset
    :param filepath: The path to the file in which to write
    '''
    with open(filepath, "w") as f:
        f.write("{} {} {} {} ".format(description.databaseType.value, description.timeLimit, description.numberOfQueries, description.numberOfTables))
        f.write(" ".join(map(lambda position : str(position), description.positionsPerTable)))
        f.write("\n")

        for i, size, id, gc, fo in dataset:
            f.write(str(i) + " " + str(size) + " " + str(id) + " ")
            f.write(" ".join(gc.to_list()) + " ")
            f.write(" ".join(fo.to_list()) + "\n")

def save_queries(
    queries : List[query.Query],
    filepath : str
):
    '''
    Saves the queries in a file, using Pickle

    :param queries: The list of queries
    :param filepath: The path to the file in which to write
    '''
    with open(filepath, "wb") as f:
        pickle.dump(queries, f)

if __name__ == "__main__":

    timelimit = 100

    fixed_database_size = False
    min_n = 40
    max_n = 70

    table_number = 2
    num_var_per_table = [3, 2]

    generator = FixedSchemaExhaustiveGenerator([(3, 2), (2, 1)])
    queries = generator.generate_queries()

    # generator = RandomGenerator([2, 2])
    # queries = generator.generate_queries(1000)

    with open(os.path.join("generated", "queries_statistics_{}_{}_{}.txt".format(table_number, len(queries), fixed_database_size)), "w") as f:
        f.write("Len {}\n".format(len(queries)))
        f.write("Total {}\n".format(generator.total_number_generated_queries))
        f.write("FO {}\n".format(generator.number_fo_rewritable_queries))
        f.write("Rejected {}\n".format(generator.number_rejected_queries))

    save_queries(queries, "generated/queries_{}_{}_{}.pickle".format(table_number, len(queries), fixed_database_size))

    for instance in [DatabaseType.YES]:
        for program in [Program.DLV]:
            description = Description(instance, timelimit, len(queries), table_number, num_var_per_table)
            if fixed_database_size:
                dataset = run_fixed_database_size(queries, instance, program, max_n, timelimit)
                save_results(description, dataset, "generated/dataset_fixed_{}_{}_{}_{}_{}_{}_{}.txt".format(program, instance.value, table_number, fixed_database_size, max_n, len(queries), timelimit))
            else:
                dataset = run_varying_database_size(queries, instance, program, min_n, max_n, timelimit)
                save_results(description, dataset, "generated/dataset_varying_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(program, instance.value, table_number, fixed_database_size, min_n, max_n, len(queries), timelimit))


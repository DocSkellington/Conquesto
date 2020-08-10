""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
from os import path
from sys import argv, stderr
import tempfile
from typing import List, Tuple
import pickle

import query.query as query
import generators.exhaustive_generators as exhaustive_generators

def load_queries(filepath : str) -> List[query.Query]:
    with open(filepath, "rb") as f:
        queries = pickle.load(f)
    return queries

if __name__ == "__main__":
    schema = [(3, 2), (2, 1)]
    # 194 is the number of queries the above schema describes
    queries_path = path.join("generated", "queries_194.pickle")

    if len(argv) == 1:
        # No arguments => generating the queries
        generator = exhaustive_generators.FixedSchemaExhaustiveGenerator(schema)
        queries = generator.generate_queries()
        # Some stats
        with open(path.join("generated", "queries_statistics_{}_{}.txt".format(len(queries), len(schema))), "w") as fi:
            fi.write("Len {}\n".format(len(queries)))
            fi.write("Total {}\n".format(generator.total_number_generated_queries))
            fi.write("FO {}\n".format(generator.number_fo_rewritable_queries))
            fi.write("Rejected {}\n".format(generator.number_rejected_queries))

        # The queries
        with open(queries_path, mode="wb") as f:
            pickle.dump(queries, f)

        print(len(queries))

    elif len(argv) == 3:
        # We generate the files, using cardinality constraints for Generate and Test
        database_type = argv[1]
        q_number = int(argv[2])
        queries = load_queries(queries_path)
        q = queries[q_number]
        print(q, file=stderr)
        tmp_path = "tmp-{}-clingo-cardinality".format(database_type)
        q.write_fo_program(path.join(tmp_path, "fo.lp"))
        q.write_generate_and_check(True, path.join(tmp_path, "gc.lp"))

    elif len(argv) == 4:
        # We generate the programs
        database_type = argv[1]
        program = argv[2]
        q_number = int(argv[3])
        queries = load_queries(queries_path)
        q = queries[q_number]
        print(q, file=stderr)
        tmp_path = "tmp-{}-{}".format(database_type, program)
        q.write_fo_program(path.join(tmp_path, "fo.lp"))
        q.write_generate_and_check(False, path.join(tmp_path, "gc.lp"))

    elif len(argv) == 5 or len(argv) == 6:
        # We generate the database files
        database_type = argv[1]
        program = argv[2]
        q_number = int(argv[3])
        alpha = int(argv[4])
        queries = load_queries(queries_path)
        q = queries[q_number]

        tmp_path = "tmp-{}-{}".format(database_type, program)
        if len(argv) == 6:
            # We use a dummy parameter to distinguish the case with cardinality constraints
            tmp_path += "-cardinality"

        # Databases
        # We print the size of the generated db
        if database_type == "no":
            print(q.write_naive_no_db(alpha, path.join(tmp_path, "no.lp")))
        elif database_type == "yes":
            with tempfile.TemporaryFile("w") as tempFile:
                target_size = q.write_naive_no_db_to_file(alpha, tempFile)
            print(q.write_naive_yes_db(alpha, path.join(tmp_path, "yes.lp"), target_size))
        elif database_type == "random":
            with tempfile.TemporaryFile("w") as tempFile:
                target_size = q.write_naive_no_db_to_file(alpha, tempFile)
            target_size = max(q.table_number, target_size)
            print(q.write_random_db(target_size, path.join(tmp_path, "random.lp")))

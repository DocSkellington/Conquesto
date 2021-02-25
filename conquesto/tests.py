from typing import List
import unittest
import tempfile
import os

import query.query
import generate_statistics

class Tests(unittest.TestCase):
    def run_query_on_naive_database(self, q: query.query.Query, db_type: generate_statistics.DatabaseType, program: generate_statistics.Program = generate_statistics.Program.CLINGO, alpha: int = 5):
        self.assertTrue(q.is_fo_rewritable())
        res = generate_statistics.run_fixed_database_size_single_query(q, db_type, generate_statistics.Program.CLINGO, alpha)
        if db_type == generate_statistics.DatabaseType.YES:
            self.assertTrue(res[0].fo.satisfiable)
            self.assertFalse(res[0].gc.satisfiable)
        elif db_type == generate_statistics.DatabaseType.NO:
            self.assertFalse(res[0].fo.satisfiable)
            self.assertTrue(res[0].gc.satisfiable)


    def run_query_on_database(self, q: query.query.Query, database: List[str], is_consistent: bool):
        self.assertTrue(q.is_fo_rewritable())
        gc_program_id, gc_program_path = tempfile.mkstemp(text=True)
        fo_program_id, fo_program_path = tempfile.mkstemp(text=True)
        db_id, db_path = tempfile.mkstemp(text=True)

        os.close(gc_program_id)
        os.close(fo_program_id)
        os.close(db_id)

        q.write_generate_and_check(False, gc_program_path)
        q.write_fo_program(fo_program_path)
        with open(db_path, "w") as db:
            for l in database:
                db.write(l + ".\n")

        fo_results = generate_statistics.launch_program(fo_program_path, db_path, generate_statistics.Program.CLINGO)
        self.assertEqual(fo_results.satisfiable, is_consistent)
        gc_results = generate_statistics.launch_program(gc_program_path, db_path, generate_statistics.Program.CLINGO)
        self.assertEqual(gc_results.satisfiable, not is_consistent)

        os.remove(gc_program_path)
        os.remove(fo_program_path)
        os.remove(db_path)


    def test_safe_variables(self):
        # The original bugs were found by Paris Koutris, Xiating Ouyang, Zhiwei Fan
        xPrim = query.query.Prim_Key("x")
        yPrim = query.query.Prim_Key("y")

        v = query.query.Variable("v")
        w = query.query.Variable("w")
        x = query.query.Variable("x")
        y = query.query.Variable("y")
        d = query.query.Variable("d")

        c = query.query.Constant("111")

        q = query.query.Query(3, [xPrim, yPrim, x, y, c, w, d], [], [[xPrim, y, c], [yPrim, x, w], [xPrim, y, d]], table_names = ['s', 'r', 't'])
        for db_type in [generate_statistics.DatabaseType.YES, generate_statistics.DatabaseType.NO]:
            with self.subTest(str(q) + ", " + str(db_type)):
                self.run_query_on_naive_database(q, db_type)

        with self.subTest(str(q) + ", special db"):
            self.run_query_on_database(q, ["s(1, 2, 111)", "r(2, 1, 222)", "t(1, 2, 333)", "t(1, 3, 444)"], False)

        q = query.query.Query(2, [xPrim, yPrim, y, c, w, v], [], [[xPrim, y, c], [yPrim, v, w]])
        for db_type in [generate_statistics.DatabaseType.YES, generate_statistics.DatabaseType.NO]:
            with self.subTest(str(q) + ", " + str(db_type)):
                self.run_query_on_naive_database(q, db_type)

        with self.subTest(str(q) + ", special db"):
            self.run_query_on_database(q, ["r(1, 2, 111)", "s(2, 3, 4)"], True)

    
    def test_free_variables(self):
        xPrim = query.query.Prim_Key("x")
        yPrim = query.query.Prim_Key("y")
        zPrim = query.query.Prim_Key("z")
        
        x = query.query.Variable("x")
        y = query.query.Variable("y")
        z = query.query.Variable("z")

        # If we remove r_3 without putting its variables as free variables, the query is not in FO
        # Otherwise, q is in FO
        q = query.query.Query(5, [xPrim, yPrim, zPrim, x, y, z], [], 
            [[xPrim, y], [yPrim, x], [xPrim, z], [zPrim, x],[yPrim, z]], 
            table_names = ['r_0', 'r_1', 'r_2', 'r_3', 'r_4'])

        self.assertTrue(q.is_fo_rewritable())

        with self.assertRaises(query.query.QueryCreationException):
            query.query.Query(4, [xPrim, yPrim, zPrim, x, y, z], [], [[xPrim, y], [yPrim, x], [xPrim, z],[yPrim, z]], table_names = ['r_0', 'r_1', 'r_2', 'r_4'], check_query=True)

        q_prime = q.remove_atom(query.query.Atom("r_3", [zPrim, x]))
        self.assertTrue(q_prime.is_fo_rewritable())


if __name__ == "__main__":
    unittest.main()
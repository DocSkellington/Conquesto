import unittest
import tempfile
import os

import query.query
import generate_statistics

class Tests(unittest.TestCase):
    def test_safe_variables(self):
        # The bug was found by Paris Koutris, Xiating Ouyang, Zhiwei Fan
        xPrim = query.query.Prim_Key("x")
        yPrim = query.query.Prim_Key("y")

        x = query.query.Variable("x")
        y = query.query.Variable("y")
        w = query.query.Variable("w")
        d = query.query.Variable("d")

        c = query.query.Constant("111")
        q = query.query.Query(3, [xPrim, yPrim, x, y, c, w, d], [], [[xPrim, y, c], [yPrim, x, w], [xPrim, y, d]], table_names = ['s', 'r', 't'])
        self.assertTrue(q.is_fo_rewritable())
        for db_type in [generate_statistics.DatabaseType.YES, generate_statistics.DatabaseType.NO]:
            res = generate_statistics.run_fixed_database_size_single_query(q, db_type, generate_statistics.Program.CLINGO, 10)
            if db_type == generate_statistics.DatabaseType.YES:
                self.assertTrue(res[0].fo.satisfiable)
                self.assertFalse(res[0].gc.satisfiable)
            elif db_type == generate_statistics.DatabaseType.NO:
                self.assertFalse(res[0].fo.satisfiable)
                self.assertTrue(res[0].gc.satisfiable)

        gc_program_id, gc_program_path = tempfile.mkstemp(text=True)
        fo_program_id, fo_program_path = tempfile.mkstemp(text=True)
        db_id, db_path = tempfile.mkstemp(text=True)

        os.close(gc_program_id)
        os.close(fo_program_id)
        os.close(db_id)

        q.write_generate_and_check(False, gc_program_path)
        q.write_fo_program(fo_program_path)

        with open(db_path, "w") as db:
            db.write("s(1, 2, 111).\n")
            db.write("r(2, 1, 222).\n")
            db.write("t(1, 2, 333).\n")
            db.write("t(1, 3, 444).")

        fo_results = generate_statistics.launch_program(fo_program_path, db_path, generate_statistics.Program.CLINGO)
        self.assertFalse(fo_results.satisfiable)
        gc_results = generate_statistics.launch_program(gc_program_path, db_path, generate_statistics.Program.CLINGO)
        self.assertTrue(gc_results.satisfiable)

        os.remove(gc_program_path)
        os.remove(fo_program_path)
        os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
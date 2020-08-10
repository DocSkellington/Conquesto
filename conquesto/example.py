""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
import query.attack_graph as attack_graph
from query.query import *

from typing import List

if __name__ == "__main__":

    x0Prim = Prim_Key("x0")
    x0 = Variable("x0")
    x1Prim = Prim_Key("x1")
    x1 = Variable("x1")
    c = Constant("c")
    cPrim = PrimaryConstant("c")

    query_ok = Query(3, [x0Prim, x0, x1Prim, c], [], [[x0Prim, x0], [x1Prim, x0], [x0Prim, c]])
    print(query_ok)
    print(query_ok.write_naive_no_db(10, "generated/no_ok.lp"))
    print(query_ok.write_naive_yes_db(10, "generated/yes_ok.lp"))
    print(query_ok.write_random_db(677))
    query_ok.write_fo_program("generated/fo_ok.lp")
    query_ok.write_generate_and_check(False, "generated/gc_ok.lp")

    x2 = Variable("x2")
    query = Query(2, [x0Prim, x1, x2], [], [[x0Prim, x1, x2], [x0Prim, x1]])
    size = query.write_naive_no_db(10)
    query.write_naive_yes_db(10, target_size=size)
    query.write_fo_program()
    query.write_generate_and_check(False)

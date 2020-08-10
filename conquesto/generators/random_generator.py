""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
from typing import List, Set, Optional, Generator
import random

import query.query as query
import generators.exhaustive_generators as exhaustive_generators

class RandomGenerator(exhaustive_generators.QueriesGenerator):
    '''
    A generator that randomly creates queries.

    The number of relations and the arity of each relation is fixed.
    '''

    def __init__(self, arities : List[int]):
        super().__init__()
        self.arities = arities

    def __iter__(self) -> Generator[query.Query, None, None]:
        while True:
            q = self._generate_query()
            if q is not None:
                yield q

    def generate_queries(self, number_queries : int) -> List[query.Query]:
        '''
        Generates the queries

        :param number_queries: The number of queries to generate

        :return: A list containing the queries generated
        '''
        self._reset_counters()

        queries : List[query.Query] = []
        while len(queries) != number_queries:
            q = self._generate_query()
            self.total_number_generated_queries += 1
            # We already know that q is FO-rewritable
            if q is not None:
                self.number_fo_rewritable_queries += 1
                if q.is_useful() and q not in queries:
                    queries.append(q)
                else:
                    self.number_rejected_queries += 1
        return queries

    def _generate_query(self) -> Optional[query.Query]:
        variables : List[query.Variable] = [query.Constant('c'), query.PrimaryConstant('c')]
        used_variables : Set[query.Variable] = set()
        tables : List[List[query.Variable]] = []
        variable_number = 0

        for arity in self.arities:
            table : List[query.Variable] = []
            primary_positions = random.randint(0, arity)
            for position in range(arity):
                # We create a new variable
                var_name = "X{}".format(variable_number)
                variable_number += 1
                variables.append(query.Variable(var_name))
                variables.append(query.Prim_Key(var_name))

                if position <= primary_positions:
                    usable_variables = list(filter(lambda var: isinstance(var, query.Prim_Key), variables))
                else:
                    usable_variables = list(filter(lambda var: not isinstance(var, query.Prim_Key), variables))

                var = random.choice(usable_variables)
                used_variables.add(var)
                table.append(var)
            
            tables.append(table)

        try:
            q = query.Query(len(tables), list(used_variables), [], tables)
            return q
        except query.QueryCreationException:
            return None
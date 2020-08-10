""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
import abc
import sys
from typing import List, Generator, Optional, Tuple
from enum import Enum
from copy import deepcopy

import query.query as query

class QueriesGenerator(abc.ABC):
    '''
    Base class of all queries generators.

    total_number_generated_queries counts the total number of generated queries.
    fo_rewritable_queries counts the number of generated FO-rewritable queries.
    rejected_queries counts the number of rejected queries. A query can be rejected because it is not FO-rewritable or it is not useful.
    These three variables are updated only in generate_queries(int).
    '''
    def __init__(self):
        self._reset_counters()

    def _reset_counters(self):
        self.number_fo_rewritable_queries = 0
        self.number_rejected_queries = 0
        self.total_number_generated_queries = 0

    @abc.abstractmethod
    def generate_queries(self, number_queries : int) -> List[query.Query]:
        pass

class FixedSchemaExhaustiveGenerator(QueriesGenerator):
    '''
    Exhaustively generate every possible query, with a fixed schema.
    A schema is the number of positions and the number of primary keys for each relation in the query.
    '''
    def __init__(self, schemas : List[Tuple[int, int]]):
        super().__init__()
        assert(len(schemas) >= 1)
        self.schemas = schemas

    def generate_queries(self, number_queries : int = -1) -> List[query.Query]:
        '''
        Generates all the possible queries for the given schema.
        Note that only queries that are FO-rewritable are returned.

        :param number_queries: Limits the number of queries to generate. If -1, there is no limit.

        :return: The list of all FO-rewritable queries generated
        '''
        self._reset_counters()
        queries : List[query.Query] = []
        gen = self._rec_generate_next_query([], [], [[]], 0)
        while number_queries == -1 or len(queries) != number_queries:
            try:
                q = next(gen)
                self.total_number_generated_queries += 1
                if q is not None:
                    self.number_fo_rewritable_queries += 1
                    queries.append(q)
                    # if q.is_useful():
                    #     queries.append(q)
                    # else:
                    #     self.rejected_queries += 1
                else:
                    self.number_rejected_queries += 1
            except StopIteration:
                # The generator is exhausted
                break
        return queries

    def __iter__(self) -> Generator[query.Query, None, None]:
        queries : List[query.Query] = []
        gen = self._rec_generate_next_query([], [], [[]], 0)
        for q in gen:
            if q is not None:
                yield q
        return None

    def _rec_generate_next_query(self,
                                 variables : List[query.Variable],
                                 variables_names : List[str],
                                 tables : List[List[query.Variable]],
                                 current_table : int
                                ) -> Generator[Optional[query.Query], None, None]:
        def add_to_tables(new_var, tables, current_table):
            new_tables = deepcopy(tables)
            new_tables[current_table].append(new_var)
            n = current_table
            if len(tables[current_table]) == self.schemas[current_table][0] - 1:
                n += 1
                new_tables.append([])
            return new_tables, n

        if current_table == len(self.schemas):
            # We have enough tables
            try:
                yield query.Query(len(self.schemas), variables, [], tables[:-1])
            except query.QueryCreationException as e:
                print(e, file=sys.stderr)
                yield None
        else:
            # We do not create a new variable
            for var in variables:
                if len(tables[current_table]) >= self.schemas[current_table][1] and isinstance(var, query.Prim_Key):
                    # We have already enough primary keys
                    continue
                elif len(tables[current_table]) < self.schemas[current_table][1] and not isinstance(var, query.Prim_Key):
                    # We do not have enough primary keys
                    continue

                new_tables, next_table = add_to_tables(var, tables, current_table)
                yield from self._rec_generate_next_query(variables, variables_names, new_tables, next_table)
            
            # We generate a new variable
            if len(tables[current_table]) < self.schemas[current_table][1]:
                # We do not yet have enough primary keys
                # We create a constant
                # We always use the same constant ('c')
                new_var : query.Variable = query.PrimaryConstant('c')
                new_tables, next_table = add_to_tables(new_var, tables, current_table)
                if new_var not in variables:
                    yield from self._rec_generate_next_query(variables + [new_var], variables_names, new_tables, next_table)

                # We then create a new variable
                # First, we reuse an already existing variable name
                for var_name in variables_names:
                    new_var = query.Prim_Key(var_name)
                    new_tables, next_table = add_to_tables(new_var, tables, current_table)
                    if new_var not in variables:
                        yield from self._rec_generate_next_query(variables + [new_var], variables_names, new_tables, next_table)
                # Second, we create a new variable name
                new_name = "x{}".format(len(variables_names))
                new_var = query.Prim_Key(new_name)
                new_tables, next_table = add_to_tables(new_var, tables, current_table)
                yield from self._rec_generate_next_query(variables + [new_var], variables_names + [new_name], new_tables, next_table)
            else:
                # We have enough primary keys
                # First, we create a constant
                # We always use the same constant ('c')
                new_var = query.Constant('c')
                new_tables, next_table = add_to_tables(new_var, tables, current_table)
                if new_var not in variables:
                    yield from self._rec_generate_next_query(variables + [new_var], variables_names, new_tables, next_table)

                # We then create a new variable
                # First, we reuse an already existing variable name
                for var_name in variables_names:
                    new_var = query.Variable(var_name)
                    new_tables, next_table = add_to_tables(new_var, tables, current_table)
                    if new_var not in variables:
                        yield from self._rec_generate_next_query(variables + [new_var], variables_names, new_tables, next_table)
                # Second, we create a new variable name
                new_name = "x{}".format(len(variables_names))
                new_var = query.Variable(new_name)
                new_tables, next_table = add_to_tables(new_var, tables, current_table)
                yield from self._rec_generate_next_query(variables + [new_var], variables_names + [new_name], new_tables, next_table)

class ExhaustiveGenerator(QueriesGenerator):
    '''
    ExhaustiveGenerator class

    Exhaustively generate every query with a fixed number of relations and number of positions in each relation.
    '''

    def __init__(self, arities : List[int]):
        '''
        Constructor of the exhaustive generator

        :param arities: The arity of each relation
        '''
        super().__init__()
        self.arities = arities

    def _next_schema(self, schema : List[Tuple[int, int]] = [], current_table : int = 0) -> Generator[List[Tuple[int, int]], None, None]:
        if current_table == len(self.arities):
            yield schema
        else:
            arity = self.arities[current_table]
            for i in range(1, self.arities[current_table] + 1):
                yield from self._next_schema(schema + [(arity, i)], current_table + 1)

    def __iter__(self) -> Generator[query.Query, None, None]:
        for schema in self._next_schema():
            gen = FixedSchemaExhaustiveGenerator(schema)
            for q in gen:
                yield q

    def generate_queries(self, number_queries : int = -1) -> List[query.Query]:
        '''
        Generates all the possible queries for the given number of relations and number of positions in each relation.
        Note that only queries that are FO-rewritable are returned.

        :param number_queries: Limits the number of queries to generate. If -1, there is no limit.

        :return: The list of all FO-rewritable queries generated
        '''
        self._reset_counters()
        schema_gen = self._next_schema()

        queries : List[query.Query] = []
        while number_queries == -1 or len(queries) < number_queries:
            try:
                schema = next(schema_gen)
                # The remaining number of queries to generate
                remaining = -1 if number_queries == -1 else number_queries - len(queries)
                gen = FixedSchemaExhaustiveGenerator(schema)
                schema_queries = gen.generate_queries(remaining)

                self.total_number_generated_queries += gen.total_number_generated_queries
                self.number_fo_rewritable_queries += gen.number_fo_rewritable_queries
                self.number_rejected_queries += gen.number_rejected_queries

                queries += schema_queries
            except StopIteration:
                # We have exhausted every schema
                break
        return queries
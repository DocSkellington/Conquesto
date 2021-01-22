""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
from __future__ import annotations
from typing import List, Set, Tuple, Dict, IO, Union
from collections import defaultdict
import multiprocessing
import os
import copy
import random
import math

import numpy as np

class QueryCreationException(Exception):
    '''
    Class for handling the exceptions that can occur when creating a query
    '''

    def __init__(self, *args):
        '''
        Constructor of an exception of creation of a query

        :params message: In *args, it contains the message we want to deliver to the user
        '''
        self.message = args[0] if args\
                               else None

    def __str__(self) -> str:
        return f"QueryCreationException : %s"%self.message if self.message is not None\
               else "QueryCreationException has been raised"

class Variable:
    '''
    Variable class

    Represents a variable of a table
    '''

    def __init__(self, name : str = "x"):
        '''
        Constructor of a Variable

        :param name: The name of the variable. Each other variable that has the same name
                    is considered as equal.
                    Note: name is case-insensitive
        '''
        self.name = name

    def __str__(self) -> str:
        return f"%s"%self.name

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.name.upper() == other.name.upper()

    def __hash__(self):
        return hash(self.name)

    def to_ASP(self) -> str:
        return self.name.upper()

class Prim_Key(Variable):
    '''
    Prim_Key class

    Represents a variable of a table that is a primary key
    '''

    def __str__(self) -> str:
        return f"PK(%s)"%self.name

    def __repr__(self) -> str:
        return str(self)

class Constant(Variable):
    '''
    Constant class

    Represents a variable of a table that is a constant
    '''
    
    def __str__(self) -> str:
        return f"CST(%s)"%self.name

    def __repr__(self) -> str:
        return str(self)

    def to_ASP(self) -> str:
        return self.name.lower()

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.name.lower() == other.name.lower()

    def __hash__(self):
        return super().__hash__()

class PrimaryConstant(Prim_Key, Constant):
    '''
    A primary key that is a constant
    '''
    
    def __str__(self) -> str:
        return f"PK_CST(%s)"%self.name

    def __repr__(self) -> str:
        return str(self)

    def to_ASP(self) -> str:
        return self.name.lower()

class Atom:
    '''
    An atom in a query
    '''
    def __init__(self, name: str, variables : List[Variable]):
        self.name : str = name
        self.variables : List[Variable] = variables

    def __str__(self) -> str:
        return self.name + "[" + ", ".join(map(str, self.variables)) + "]"
        
    def __repr__(self) -> str:
        return str(self)

    def to_ASP(self) -> str:
        return self.name.lower() + "(" + ", ".join(map(lambda x: x.to_ASP(), self.variables)) + ")"

    @property
    def pure_variables(self) -> Set[Variable]:
        '''
        Gets all variables as instances of Variable.

        That is, the primary key aspect is ignored.
        However, constants are removed
        '''
        s : Set[Variable] = set()
        for x in self.variables:
            if not isinstance(x, Constant):
                s.add(Variable(x.name))
        return s

    @property
    def primary_variables(self) -> Tuple[Prim_Key, ...]:
        return tuple([x for x in self.variables if isinstance(x, Prim_Key)])

    def primary_variables_as_non_primary(self) -> Tuple[Variable, ...]:
        '''
        Creates a tuple with the primary variables cast as non-primary.
        That is, primary constants are cast to constants and non-constants to variables.
        :return: A tuple with the primary variables cast as simple variables or constants.
        '''
        l : List[Variable] = []
        for var in self.primary_variables:
            if isinstance(var, PrimaryConstant):
                l.append(Constant(var.name))
            elif isinstance(var, Prim_Key):
                l.append(Variable(var.name))
        return tuple(l)

    @property
    def secondary_variables(self) -> Tuple[Variable, ...]:
        return tuple([x for x in self.variables if not isinstance(x, Prim_Key)])

class Query:
    '''
    Query class

    Represents a query (always with no free variables)

    A query can not be empty.
    '''

    def __init__(self,
                table_number : int,
                variables : List[Variable],
                free_variables : List[Variable],
                table_attributes : List[List[Variable]],
                table_names : List[str] = [],
                check_query = True):
        '''
        Constructor of a query

        :param table_number: The number of tables contained in the query
        :param variables: The list of the variables of the query
        :param table_attributes: The number of attributes of each table
                                Example : one table and 5 variables, we have [[x_1, x_2, x_3, x_4, x_5]] where
                                    each x_i is an instance of the class Variable (or child)
        :param free_variables: The list of free variables in the query
        :param check_query: True if the query has to be checked (for example, if the query is handwritten
                            and the author is not sure having made no errors)
        '''

        self.table_number : int = table_number
        if self.table_number < 0:
            raise QueryCreationException("A query must contain at least one atom")

        # If/else to avoid mistakes from the user
        self.variables : List[Variable] = variables
        self.table_attributes : List[List[Variable]] = table_attributes
        # We have a maximum of 26 different tables
        # We begin to the name "R"
        # It's only useful to print things and make the ASP programs with clear names
        if len(table_names) == 0:
            self.table_names : List[str] = [chr(65 + (17 + i) % 26)\
                                            for i in range(self.table_number)]
        else:
            self.table_names = table_names

        self.atoms = [Atom(self.table_names[i], self.table_attributes[i])\
                                for i in range(self.table_number)]

        self.free_variables = free_variables

        check_query = check_query and self.table_number > 0
        if check_query:
            self.check_query()

    def __str__(self) -> str:
        return "The query is empty" if self.table_number == 0\
                                    else self._construct_str()

    def __getstate__(self):
        return self.table_attributes, self.table_names, self.table_number, self.variables, self.free_variables, self.atoms

    def __setstate__(self, d):
        self.table_attributes, self.table_names, self.table_number, self.variables, self.free_variables, self.atoms = d

    def __eq__(self, other) -> bool:
        # Two queries are equal if it exists an homomorphism from self to other and an homomorphism from other to self
        if not isinstance(other, Query):
            return False

        if self.table_attributes is None:
            return True if other.table_attributes is None\
                        else False

        if not len(self.table_attributes) == len(other.table_attributes):
            return False

        return self._has_homomorphism(other) and other._has_homomorphism(self)

    def _has_homomorphism(self, other: Query) -> bool:
        class RunHomomorphism:
            '''
            Class used to parallelize the homomorphism's computations
            '''
            def __init__(self, base: Query, other: Query):
                self.base = base
                self.other = other

            def __call__(self, homomorphism):
                self.base._check_homomorphism(self.other, homomorphism)

        # We generate every possible homomorphism and checks if it is valid
        gen = self._generate_homomorphism(self.variables, other.variables, {})
        homomorphisms = list(gen)

        # We parallelize the search of a valid homomorphism
        pool = multiprocessing.Pool(8)
        reslist = pool.imap_unordered(RunHomomorphism(self, other), homomorphisms)
        pool.close()

        result = False
        for res in reslist:
            if res:
                pool.terminate()
                result = True
                break
        pool.join()
        return result

    def _generate_homomorphism(self, variables_self: List[Variable], variables_other: List[Variable], homomorphism: Dict[str, str]):
        if len(variables_self) == 0:
            yield homomorphism
        else:
            var_self = variables_self[0]
            remaining_self = variables_self[1:] if len(variables_self) > 0 else []
            for j, var_other in enumerate(variables_other):
                homomorphism[var_self.name] = var_other.name
                remaining_other = variables_other[:j]
                if j != len(variables_other) - 1:
                    variables_other[j+1:]
                yield from self._generate_homomorphism(remaining_self, remaining_other, homomorphism)

    def _check_homomorphism(self, other: Query, homomorphism : Dict[str, str]) -> bool:
        tables_other = copy.deepcopy(other.table_attributes)

        for table_self in self.table_attributes:
            table_other : List[Variable] = []
            for var in table_self:
                if isinstance(homomorphism[var.name], Constant) != isinstance(var, Constant):
                    return False
                if isinstance(homomorphism[var.name], Prim_Key) != isinstance(var, Prim_Key):
                    return False

                homo_name = homomorphism[var.name]
                if isinstance(var, PrimaryConstant):
                    table_other.append(PrimaryConstant(homo_name))
                elif isinstance(var, Constant):
                    table_other.append(Constant(homo_name))
                elif isinstance(var, Prim_Key):
                    table_other.append(Prim_Key(homo_name))
                else:
                    table_other.append(Variable(homo_name))

            if table_other not in tables_other:
                return False
            tables_other.remove(table_other)

        return True

    def _construct_str(self) -> str:
        '''
        Constructs the string to print when there is at least one table
        '''
        str_repr : str = ""
        for i, atom in enumerate(self.atoms):
            str_repr += str(atom)
            str_repr += " AND " if i + 1 < len(self.atoms) else ""

        return str_repr

    def check_query(self):
        '''
        Ensures that the parameters given to this class are good to make a consistent and FO-rewritable query
        '''
        # We don't want to have a table number different of the real number of tables
        
        if self.table_number != len(self.table_attributes):
            raise QueryCreationException("the number of tables given is different from the real number of tables")

        # We don't want a variable that is a constant in a table
        # and a classic variable in another table

        for var in self.variables:
            if isinstance(var, Constant):
                for other in [val for sublist in self.table_attributes for val in sublist]:
                    if var == other and not isinstance(other, Constant):
                        raise QueryCreationException(f"The variable \"%s\" occurs either as a constant and as a non constant variable"%var.name)
        
        # For readability, we want to have all the primary keys first

        for table in self.table_attributes:

            no_more_pk : bool = False

            for var in table:
                if not isinstance(var, Prim_Key):
                    no_more_pk = True
                elif no_more_pk and isinstance(var, Prim_Key):
                    raise QueryCreationException(f"The primary key \"%s\" is found after a non primary key variable"%var.name)

        if not self.is_fo_rewritable():
            raise QueryCreationException("The query {} is not FO-rewritable".format(self))

        return

    def is_fo_rewritable(self) -> bool:
        '''
        Is the query rewritable in FO?

        :return: True iff the query is rewritable in FO
        '''
        import query.attack_graph as attack_graph
        ag = attack_graph.AttackGraph(self)
        return ag.is_acyclic

    def is_useful(self) -> bool:
        '''
        Return if a query is useful (at least not useless) or not.

        We know that a query is useless if we do not have one of the two case below :
            - A variable X is used at least twice and on different tables.
            - There is a constant in the query
        '''

        seen_variables_all_tables : List[Variable] = []
        for i, table in enumerate(self.table_attributes):
            seen_variables_this_table : List[Variable] = []
            for _, var in enumerate(table):
                if not isinstance(var, Constant):
                    if not var in seen_variables_all_tables:
                        seen_variables_all_tables.append(var)
                        seen_variables_this_table.append(var)
                    else:
                        #we have seen a variable at least two times and on different tables, the query is not considered as useless
                        if not var in seen_variables_this_table:
                            return True
                else:
                    #we have seen a constant, the query is not considered as useless
                    return True

        #we have not encountered one of the two case, the query is considered as useless.
        return False

    def write_generate_and_check_to_file(self, use_cardinality_constraint : bool, f : IO) -> bool:
        '''
        Writes the query into a generate and check ASP program

        :param f: The file-like object in which to write

        :return: True if the program has successfully been written
        '''
        # Generate part
        f.write("% Generating part\n\n")
        if use_cardinality_constraint:
            for i, table in enumerate(self.table_attributes):
                code_line : str = "{" +\
                                    self.table_names[i].lower() +\
                                    "r" +\
                                    "("

                pk_num : int = 0
                while pk_num < len(table) and isinstance(table[pk_num], Prim_Key):
                    pk_num += 1
                
                params : str = ""
                for j, var in enumerate(table):
                    name : str = "A" + str(j)
                    params += name + ", " if j + 1 < len(table) else name

                code_line += params + ") : " + self.table_names[i].lower() + "(" + params + ")} == 1 :- " +\
                                self.table_names[i].lower() + "("

                for j, var in enumerate(table):
                    name = "A" + str(j)
                    if j < pk_num:
                        code_line += name + ", " if j + 1 < len(table) else name + ").\n"
                    else:
                        code_line += "_, " if j + 1 < len(table) else "_).\n"

                f.write(code_line)

        else: # not use_cardinality_constraint
            for i, table in enumerate(self.table_attributes):
                table_name = self.table_names[i].lower()
                pk_num = 0
                while pk_num < len(table) and isinstance(table[pk_num], Prim_Key):
                    pk_num += 1
                
                params : List[str] = ["A" + str(j) for j in range(len(table))]
                params_whole = ", ".join(params)
                params_primary = ", ".join(params[:pk_num])

                # We select which fact is in the repair and which one is deleted
                f.write(table_name + "r(" + params_whole + ") :- " + table_name + "(" + params_whole + "), not delete_" + table_name + "(" + params_whole + ").\n")
                f.write("delete_" + table_name + "(" + params_whole + ") :- " + table_name + "(" + params_whole + "), not " + table_name + "r(" + params_whole + ").\n")
                # We enforce that each block contains at least one tuple
                # We do not need to check that each block contains at most one tuple.
                # Indeed, if a falsifying repair exists, then the ASP solver will output a superset of that repair
                f.write("repaired_block_" + table_name + "(" + params_primary + ") :- " + table_name + "r(" + params_whole + ").\n")
                f.write(":- " + table_name + "(" + params_whole + "), not repaired_block_" + table_name + "(" + params_primary + ").\n")
                f.write("\n")
        # Check part

        f.write("\n% Check part\n\n")

        code_line = ":- "

        var_names : List[str] = []
        for i, var in enumerate(self.variables):
            var_names.append(var.name.lower() if isinstance(var, Constant) else var.name.upper())
        
        for i, table in enumerate(self.table_attributes):
            code_line += self.table_names[i].lower() + "r" + "("

            for j, var in enumerate(table):
                name = var_names[self.variables.index(var)]
                code_line += name
                code_line += ", " if j + 1 < len(table) else ")"
            
            code_line += ", " if i + 1 < self.table_number else "."
        f.write(code_line)
        
        return True

    def write_generate_and_check(self, use_cardinality_constraint : bool, f_name : str = os.path.join("generated", "gc.lp")) -> bool:
        '''
        Writes the query into a generate and check ASP program

        :param f_name: The name of the file where to write the program

        :return: True if the program has successfully been written
        '''
        with open(f_name, "w") as f:
            r = self.write_generate_and_check_to_file(use_cardinality_constraint, f)
        return r

    def write_fo_program_to_file(self, f: IO) -> bool:
        '''
        Writes the query into a FO ASP program, using our rewriting.

        :param f: The file-like object in which to write

        :return: True if the program has successfully been written
        '''
        import query.fo_rewriting as fo_rewriting
        tree = fo_rewriting.fo_rewriting(self)
        if tree is None:
            return False
        print(tree.to_ASP("alpha_1"))
        f.write(tree.to_ASP("alpha_1") + "\n" + ":- not alpha_1_1.")
        return True

    def write_fo_program(self, f_name: str = os.path.join("generated", "fo.lp")) -> bool:
        '''
        Writes the query into a FO ASP program, using our rewriting.

        :param f_name: The name of the file where to write the program

        :return: True if the program has successfully been written
        '''
        with open(f_name, "w") as f:
            r = self.write_fo_program_to_file(f)
        return r

    def write_naive_yes_db_to_file(self, alpha : int, f : IO, target_size : int = -1):
        '''
        Writes a naive yes (in the sense that CERTAINTY(q) is always true) database in ASP for this query

        :param alpha: The number of values a variable can have
        :param f: The file-like object in which to write
        :param target_size: If not -1, the function tries to create a database whose size is as close as possible to the given value. It can be used to try to have same sizes for yes and no databases.

        :return: The size of the database if it has successfully been written
        '''
        # For each table, we construct a dictionary mapping each variable to its values
        # If the variable is a constant, then it is mapped to that constant
        # Otherwise, it is mapped to the lower and upper bounds.
        assigned_values_per_table : List[Dict[str, Union[str, Tuple[int, int]]]] = []

        def database_size(assigned_values_per_table):
            '''
            Gives the size of the database, based on the current assigned values
            '''
            size = 0
            for i in range(self.table_number):
                table_size = 1
                for value in assigned_values_per_table[i].values():
                    if isinstance(value, tuple):
                        lower, upper = value
                        table_size *= upper - lower + 1
                size += table_size
            return size

        # We assign the values, using alpha
        for i, table in enumerate(self.table_attributes):
            assigned_values : Dict[str, Union[str, Tuple[int, int]]] = {}
            for var in table:
                var_name = var.name.upper()
                # We do not re-assign a value to an already seen variable
                if var_name in assigned_values:
                    continue
                if isinstance(var, Constant):
                    assigned_values[var_name] = var.name.lower()
                elif isinstance(var, Prim_Key):
                    assigned_values[var_name] = (1, alpha)
                else:
                    assigned_values[var_name] = (1, 1)
            assigned_values_per_table.append(assigned_values)

        # It may happen that the above loop did not generate enough tuples
        # In this case, we continue on adding new ones
        change = True
        while change and database_size(assigned_values_per_table) < target_size:
            change = False
            for i, table in enumerate(self.table_attributes):
                seen_variables : List[str] = []
                for var in table:
                    # We do not touch constants not already-seen variables
                    # Furthermore, the variable must be primary
                    var_name = var.name.upper()
                    if not isinstance(var, Constant) and isinstance(var, Prim_Key) and not var_name in seen_variables:
                        seen_variables.append(var_name)
                        lower, upper = assigned_values_per_table[i][var_name]
                        assigned_values_per_table[i][var_name] = (lower, upper + alpha)
                        change = True

        # Finally, we can write the actual database
        # We create a generator to help us
        def generate_next_tuple(table_name, table, assigned_values, current_values, i):
            '''
            Using the given assigned values for a single table, it generates each tuple to put in the database
            '''
            if len(current_values) == len(assigned_values):
                yield table_name.lower() + "(" + ", ".join(map(lambda var: str(current_values[var.name.upper()]), table)) + ").\n"
            else:
                var_name = list(assigned_values.keys())[i]
                value = assigned_values[var_name]
                if isinstance(value, tuple):
                    lower, upper = value
                    for val in range(lower, upper + 1):
                        current_values[var_name] = val
                        yield from generate_next_tuple(table_name, table, assigned_values, current_values, i + 1)
                        del current_values[var_name]
                else:
                    current_values[var_name] = value
                    yield from generate_next_tuple(table_name, table, assigned_values, current_values, i + 1)
                    del current_values[var_name]

        # We write the db to the file, using the generator
        for i, table in enumerate(self.table_attributes):
            table_name = self.table_names[i]
            for line in generate_next_tuple(table_name, table, assigned_values_per_table[i], {}, 0):
                f.write(line)

        return database_size(assigned_values_per_table)
    
    def write_naive_yes_db(self, alpha : int, f_name : str = os.path.join("generated", "naive_yes_db.lp"), target_size : int = -1) -> int:
        '''
        Writes a naive yes (in the sense that CERTAINTY(q) is always true) database in ASP for this query

        :param alpha: The number of values a variable can have
        :param f_name: The name of the file where to write the program
        :param target_size: If not -1, the function tries to create a database whose size is as close as possible to the given value. It can be used to try to have same sizes for yes and no databases.

        :return: The size of the database if it has successfully been written
        '''
        with open(f_name, "w") as f:
            r = self.write_naive_yes_db_to_file(alpha, f, target_size)
        return r

    def write_naive_no_db_to_file(self, alpha : int, f : IO) -> int:
        '''
        Writes a naive no (in the sense that CERTAINTY(q) is always false) database in ASP for this query

        :param alpha: The maximum number of values a variable can have
        :param f: The file-like object in which to write

        :return: The size of the database if it has successfully been written
        '''
        
        database_size = 0
        number_occurrences : Dict[str, int] = defaultdict(int)
        at_least_one_constant = False
        assigned_values_per_table : List[List[Union[str, Tuple[int, int]]]] = []

        # We start by computing the assignments of each variable
        # If a relation has a constant, we change the name of the constant in the db. Thus, the whole query becomes unsatisfiable.
        # If a variable appears multiple times in the query, then we make sure different values are assigned to each occurrence
        # Otherwise, the empty database is the only possibility
        for i, table in enumerate(self.table_attributes):
            seen_variables_this_table : List[str] = []
            table_name = self.table_names[i].lower()
            table_size = 1
            assigned_values : List[Union[str, Tuple[int, int]]] = []

            for var in table:
                if isinstance(var, Constant):
                    at_least_one_constant = True
                    assigned_values.append(var.name.lower()*2)
                else:
                    var_name = var.name.upper()
                    lower = number_occurrences[var_name] * alpha + 1
                    upper = lower + alpha - 1
                    assigned_values.append((lower, upper))
                    number_occurrences[var_name] += 1
                    table_size *= alpha

            assigned_values_per_table.append(assigned_values)
            database_size += table_size
        
        # We check if the database should be empty
        if not at_least_one_constant:
            # There is no constant in the relation. We must then check if a variable appeared multiple times
            if len(list(filter(lambda val: val > 1, number_occurrences.values()))) == 0:
                # There is no such variable
                # Thus, the database must be empty
                f.write("\n")
                return 0
            
        # Then, we create the tuples to put in the database
        def generate_next_tuple(table_name, assigned_values, current_values):
            if len(current_values) == len(assigned_values):
                yield table_name.lower() + "(" + ", ".join(map(str, current_values)) + ").\n"
            else:
                i = len(current_values)
                value = assigned_values[i]
                if isinstance(value, tuple):
                    lower, upper = value
                    for val in range(lower, upper + 1):
                        yield from generate_next_tuple(table_name, assigned_values, current_values + [val])
                else:
                    yield from generate_next_tuple(table_name, assigned_values, current_values + [value])
                    
        for i in range(self.table_number):
            table_name = self.table_names[i].lower()
            for line in generate_next_tuple(table_name, assigned_values_per_table[i], []):
                f.write(line)

        return database_size

    def write_naive_no_db(self, alpha : int, f_name : str = os.path.join("generated",  "naive_no_db.lp")) -> int:
        '''
        Writes a naive no (in the sense that CERTAINTY(q) is always false) database in ASP for this query

        :param alpha: The maximum number of values a variable can have
        :param f_name: The name of the file where to write the program

        :return: The size of the database if it has successfully been written
        '''
        with open(f_name, "w") as f:
            r = self.write_naive_no_db_to_file(alpha, f)
        return r

    def write_random_db_to_file(self, target_size : int, f : IO) -> int:
        '''
        Writes a random database (in the sense that CERTAINTY(q) can be true or false) in ASP.
        If an atom in the query contains a constant, the database can also contain this constant.

        Each random sampling is done using an uniform distribution.

        :param target_size: The wanted size of the database. The database has exactly the requested size.
        :param f: The file-like object in which to write the database.

        :return: The size of the database.
        '''
        assert target_size >= self.table_number, "The target size (received: {}) of the random database must be at least the number of atoms in the query (here, {})".format(target_size, self.table_number)

        database_size = 0
        remaining_target_size = target_size
        rng = np.random.default_rng()
        higher_value = target_size

        target_table_size = math.floor(target_size / self.table_number)

        constant_names = list(map(lambda const: const.name.lower(), filter(lambda var: isinstance(var, Constant), self.variables)))

        for i, table in enumerate(self.table_attributes):
            seen_variables_names_this_table : List[str] = []
            t_name = self.table_names[i].lower()
            number_occurrences_this_table : Dict[str, int] = defaultdict(int)

            if i == self.table_number - 1 and target_table_size * self.table_number == target_size - 1:
                target_table_size += 1

            for j in range(target_table_size):
                code_line = t_name + "("
                for k in range(len(table)):
                    # Do we pick a name among the constants or a random integer?
                    if len(constant_names) > 0 and rng.random() < .5:
                        # We pick a constant name
                        value = rng.choice(constant_names)
                    else:
                        value = rng.integers(1, higher_value, endpoint = True)
                    if k != 0:
                        code_line += ", "
                    code_line += str(value)
                code_line += ").\n"
                f.write(code_line)
                database_size += 1

            remaining_target_size = target_size - database_size

        return database_size

    def write_random_db(self, target_size : int, f_name : str = os.path.join("generated", "random_db.lp")):
        '''
        Writes a random database (in the sense that CERTAINTY(q) can be true or false) in ASP.
        If an atom in the query contains a constant, the database can also contain this constant.

        Each random sampling is done using an uniform distribution.

        :param target_size: The wanted size of the database. The database has exactly the requested size.
        :param f_name: The name of the file in which to write.

        :return: The size of the database.
        '''
        with open(f_name, "w") as f:
            r = self.write_random_db_to_file(target_size, f)
        return r

    @property
    def pure_variables(self) -> Set[Variable]:
        '''
        Gets all variables as instances of Variable.

        That is, the primary key aspect is ignored.
        However, constants are removed
        :return: the set of variables
        '''
        s : Set[Variable] = set()
        for x in self.variables:
            if not isinstance(x, Constant):
                s.add(Variable(x.name))
        return s

    def remove_atom(self, atom : Atom) -> Query:
        '''
        Constructs a copy of this query without the given atom.
        It also sets the free variables accordingly.

        :param atom: The atom to remove
        :return: The query without the atom (but with free variables)
        '''
        atoms_to_keep = list(filter(lambda x: atom.name != x.name, self.atoms))
        # variables_in_other_atoms = set([var for x in self.atoms for var in x.variables if x != atom])
        # free_variables = set(atom.variables).union(self.free_variables).intersection(variables_in_other_atoms)
        free_variables = set(atom.pure_variables).union(self.free_variables)
        new_query = Query(self.table_number - 1, self.variables, list(free_variables), [x.variables for x in atoms_to_keep], table_names=[x.name for x in atoms_to_keep])
        return new_query

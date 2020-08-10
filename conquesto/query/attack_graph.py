""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
from typing import List, Set, Dict
from collections import namedtuple, defaultdict
import copy

import query.query as query

class PrimaryKeyConstraint(namedtuple("PrimaryKeyConstraint", ["primary", "secondary"])):
    '''
    A primary key constraint.

    It simply contains a tuple with the "primary variables" and a tuple with the "secondary variables"
    '''
    def __str__(self) -> str:
        return "(" + ", ".join(map(str, self.primary)) + ") -> " + str(self.secondary)

class PrimaryKeyConstraintsSet:
    '''
    A set with all primary key constraints.

    It can compute the closure of itself
    '''
    def __init__(self):
        self.constraints : Set[PrimaryKeyConstraint] = set()
    
    def __str__(self) -> str:
        return "{" + ", ".join(map(str, self.constraints)) + "}"

    def add_from_atom(self, atom : query.Atom):
        primary = atom.primary_variables_as_non_primary()
        if len(primary) > 0:
            secondary = atom.secondary_variables
            # If primary is composed of only constants, then the actual primary key is the empty set
            primary_without_constants = tuple(filter(lambda var: not isinstance(var, query.Constant), primary))

            # We want only constraints of shape (x, y, z, ...) -> w
            for var in secondary:
                self.constraints.add(PrimaryKeyConstraint(primary_without_constants, var))

    def satisfies(self, constraint: PrimaryKeyConstraint) -> bool:
        '''
        Does this set always satisfied the given constraint?
        '''
        primary, secondary = constraint

        if secondary in primary:
            return True
        
        # See Algorithme 1 on page 38 of http://informatique.umons.ac.be/ssi/teaching/fbd/theorie/syllabus.pdf
        unused = set(self.constraints)
        closure = set(primary)
        changed = True
        while changed:
            changed = False

            to_remove : Set[PrimaryKeyConstraint] = set()
            for W, Z in unused:
                if closure.issuperset(W):
                    to_remove.add(PrimaryKeyConstraint(W, Z))
                    closure.update([Z])

            if len(to_remove) != 0:
                unused.difference_update(to_remove)
                changed = True
                
        return secondary in closure

class AttackGraph:
    '''
    An attack graph

    See "Consistent Query Answering for Primary Keys" by Koutris and Wijsen
    '''
    def __init__(self, q : query.Query):
        self.query = q
        self.atoms : List[query.Atom] = q.atoms
        self.attacks : Dict[query.Atom, Set[query.Atom]] = defaultdict(set)
        for atom in self.atoms:
            self._compute_attacks_from(atom)
        self.acyclic = True
        for atom in self.atoms:
            for intermediate in self.attacks[atom]:
                if atom in self.attacks[intermediate]:
                    self.acyclic = False
                    break
            if not self.acyclic:
                break

    def __str__(self) -> str:
        s = "; ".join(map(lambda s: s[0].name + " attacks " + ", ".join(map(lambda x: x.name, s[1])), self.attacks.items()))
        return s

    def _compute_attacks_from(self, atom):
        '''
        Computes every attack from the given atom
        '''
        F = self._compute_F(atom)
        for a in self.atoms:
            if a != atom:
                attacking_vars = a.pure_variables.intersection(atom.pure_variables).difference(F)
                if len(attacking_vars) != 0:
                    self.attacks[atom].add(a)
                    self._compute_attacks_from_recursive(atom, F, a, set([atom, a]))

    def _compute_attacks_from_recursive(self, start: query.Atom, F: Set[query.Variable], previous: query.Atom, visited: Set[query.Atom]):
        for a in self.atoms:
            if a not in visited:
                attacking_vars = a.pure_variables.intersection(previous.pure_variables).difference(F)
                if len(attacking_vars) != 0:
                    self.attacks[start].add(a)
                    self._compute_attacks_from_recursive(start, F, a, visited.union(set([a])))

    def _compute_F(self, removed_atom: query.Atom) -> Set[query.Variable]:
        '''
        Computes the set F as described in the reference paper
        '''
        atoms = [a for a in self.atoms if a != removed_atom]
        primary_key_constraints = PrimaryKeyConstraintsSet()
        for atom in atoms:
            primary_key_constraints.add_from_atom(atom)

        primary_atom = removed_atom.primary_variables_as_non_primary()
        F : Set[query.Variable] = set()
        variables = self.query.pure_variables
        for x in variables:
            if primary_key_constraints.satisfies(PrimaryKeyConstraint(primary_atom, x)):
                F.add(x)
        return F

    @property
    def number_of_atoms(self) -> int:
        return len(self.atoms)

    @property
    def unattacked_atoms(self) -> List[query.Atom]:
        unattacked: List[query.Atom] = []
        for atom in self.atoms:
            is_attacked = False
            for a in self.atoms:
                if atom in self.attacks[a]:
                    is_attacked = True
                    break

            if not is_attacked:
                unattacked.append(atom)
        return unattacked

    @property
    def is_acyclic(self):
        '''
        Is this attack graph acyclic?
        '''
        return self.acyclic
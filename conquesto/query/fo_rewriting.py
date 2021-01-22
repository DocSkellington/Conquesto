""" Copyright © - 2020 - UMONS
    CONQUESTO of University of Mons - Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet - is free software : you can redistribute it and/or modify it under the terms of the BSD-3 Clause license. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the BSD-3 Clause License for more details. 
    You should have received a copy of the BSD-3 Clause License along with this program. 
    Each use of this software must be attributed to University of Mons (Jonathan Joertz, Dorian Labeeuw, and Gaëtan Staquet).
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Set
from abc import ABC, abstractmethod
import copy

import query.attack_graph as attack_graph
import query.query as query

def next_subtree_head_name(head_name : str) -> str:
    '''
    Generate the head name to use in the next tree.

    :param head_name: of shape alpha_1
    :return: of shape alpha_2
    '''
    parts = head_name.split(sep = "_")
    parts[1] = str(int(parts[1]) + 1)
    return parts[0] + "_" + parts[1]

def next_node_head_name(head_name : str) -> str:
    '''
    Generate the next head name to use (without changing the tree).

    :param head_name: of shape alpha_1_1
    :return: of shape alpha_1_2
    '''
    parts = head_name.split(sep = "_")
    parts[2] = str(int(parts[2]) + 1)
    return "_".join(parts)

class Node(ABC):
    def __init__(self, children: List[Node]):
        self.children = children

    def set_children(self, children: List[Node]):
        self.children = children

    @abstractmethod
    def apply_negation(self) -> Node:
        '''
        Applies the negation on the current node and returns the new node (with the negation applied).
        '''
        pass

    @abstractmethod
    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        '''
        Writes this node in an ASP program in a rule of the given name 

        :param head_name: The name of the current rule we are writing
        :param free_variables: The list of free variables in the current rule
        :param for_save_variables: The list of atoms that can be used to ensure that every free variable is safe
        :param unsafe_variables: The set of unsafe variables
        :return: The string giving the ASP program
        '''
        pass

    @abstractmethod
    def _get_primary_keys(self) -> List[query.Variable]:
        pass

    def _apply_negation_on_children(self) -> List[Node]:
        return [child.apply_negation() for child in self.children]

    def _add_atoms_for_unsafe_variables(self, for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        '''
        Creates a string with every needed atom in order to make every variable safe.
        :param for_save_variables: The atoms that can be used to make variables safe.
        :param unsafe_variables: The unsafe variables.
        :return: The part of an ASP body with atoms to make sure every variable is safe.
        '''
        s = ""

        # We need to first iterate over the atoms to be able to remove the variables from the set
        for for_safe_atom in for_safe_variables:
            contains_var = False
            # We check if the current atom is useful
            for unsafe_var in unsafe_variables:
                contains_var = len(list(filter(lambda v: v.name == unsafe_var.name, for_safe_atom.pure_variables))) != 0
                if contains_var:
                    break
            # If it is useful, we can use it to make some variables safe
            if contains_var:
                # In order to properly make the variables safe, we create new atoms
                # Each atom is a copy of the current useful atom where all the variables' names are changed except for one of the unsafe variables
                # For instance, if the atom R[X, Y, Z] can be used to make the variables [X, Z] safe, we create two new atoms
                # R[X, YYY_0_1, ZZZ_0_2]
                # R[XXX_1_0, YYY_1_1, Z]
                for i, variable in enumerate(for_safe_atom.variables):
                    # If the current variable is already safe, we skip it
                    if variable not in unsafe_variables:
                        continue
                    variables: List[query.Variable] = []
                    unsafe_variables.remove(variable) # We mark the variable as being safe
                    for j, var in enumerate(for_safe_atom.variables):
                        if i == j:
                            # We are at the variable we want to make safe
                            variables += [variable]
                        elif isinstance(var, query.Constant):
                            # We do not change constants
                            variables += [var]
                        else:
                            # We change the variable's name
                            new_var = copy.copy(var)
                            new_var.name = (3 * var.name) + "_{}_{}".format(i, j)
                            variables += [new_var]
                    # Now that we have all the variables' names, we can create the atom and convert it to ASP
                    new_atom = query.Atom(for_safe_atom.name, variables)
                    s += new_atom.to_ASP() + ", "

        return s

class NotNode(Node):
    def apply_negation(self) -> Node:
        return NotNode(self._apply_negation_on_children())

    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        next_head_name = next_node_head_name(head_name)
        primary_keys = self.children[0]._get_primary_keys()

        new_unsafe_variables = set(primary_keys + free_variables)
        new_free_variables = list(new_unsafe_variables)

        if len(new_free_variables) != 0:
            in_parenthesis = "(" + ", ".join(map(lambda var: var.to_ASP(), new_free_variables)) + ")"
            s = "not " + next_head_name + in_parenthesis + ".\n"
            s += next_head_name + in_parenthesis 
        else:
            s = "not " + next_head_name + ".\n"
            s += next_head_name
        s += " :- " + self.children[0].to_ASP(next_head_name, new_free_variables, for_safe_variables, new_unsafe_variables)
        return s

    def _get_primary_keys(self) -> List[query.Variable]:
        return []

    def __str__(self) -> str:
        return "Not (" + str(self.children[0]) + ")"

class ExistsNode(Node):
    def apply_negation(self) -> Node:
        raise NotImplementedError("Impossible to negate an Exists node in this context.")

    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        return self.children[0].to_ASP(head_name, free_variables, for_safe_variables, unsafe_variables)

    def _get_primary_keys(self) -> List[query.Variable]:
        return self.children[0]._get_primary_keys()

    def __str__(self) -> str:
        return "Exists (" + str(self.children[0]) + ")"

    def descend_into_OR(self) -> OrNode:
        if not isinstance(self.children[0], OrNode):
            raise RuntimeError("Your construction is not correct")
        return OrNode([ExistsNode([x]) for x in self.children[0].children])

class AndNode(Node):
    def apply_negation(self) -> OrNode:
        return OrNode(self._apply_negation_on_children())

    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        return ", ".join(map(lambda child: child.to_ASP(head_name, free_variables, for_safe_variables, unsafe_variables), self.children))

    def _get_primary_keys(self) -> List[query.Variable]:
        # The query atom must always be in the left part of the conjunction
        return self.children[0]._get_primary_keys()

    def __str__(self) -> str:
        return "(" + ") AND (".join(map(str, self.children)) + ")"

    def distribute_over_or(self) -> OrNode:
        if not isinstance(self.children[1], OrNode):
            raise RuntimeError("Your construction is incorrect")
        left = self.children[0]
        and_nodes : List[Node] = []
        for right in self.children[1].children:
            and_nodes.append(AndNode([left, right]))
        return OrNode(and_nodes)

class OrNode(Node):
    def apply_negation(self) -> AndNode:
        return AndNode(self._apply_negation_on_children())

    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        in_parenthesis = "(" + ", ".join(map(lambda x: x.to_ASP(), free_variables)) + ")"

        new_unsafe_variables = unsafe_variables.union(free_variables)

        s = ""
        # We now, by construction, that the parent of this node is a NotNode
        # Thus, we also know that the previous body is already closed.
        # So, we can just create the new rules
        s += self.children[0].to_ASP(head_name, free_variables, for_safe_variables, copy.deepcopy(new_unsafe_variables))
        for child in self.children[1:]:
            s += head_name + in_parenthesis + " :- " + child.to_ASP(head_name, free_variables, for_safe_variables, copy.deepcopy(new_unsafe_variables))
        return s

    def _get_primary_keys(self) -> List[query.Variable]:
        return self.children[0]._get_primary_keys()

    def __str__(self) -> str:
        return "(" + ") OR (".join(map(str, self.children)) + ")"

class ImpliesNode(Node):
    def apply_negation(self) -> AndNode:
        return AndNode([self.children[0], self.children[1].apply_negation()])

    def _get_primary_keys(self) -> List[query.Variable]:
        raise NotImplementedError("An Implies node should not be present")

    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        raise NotImplementedError("An Implies node should not be present")

    def __str__(self) -> str:
        return "(" + ") IMPLIES (".join(map(str, self.children)) + ")"

class Leaf(Node):
    def __init__(self, negation : bool = False):
        self.negation = negation

class AtomNode(Leaf):
    def __init__(self, atom : query.Atom, negation : bool = False):
        super().__init__(negation)
        self.atom = atom

    def apply_negation(self) -> AtomNode:
        return AtomNode(self.atom, not self.negation)

    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        # We update the set of unsafe variables to remove variables that are safe thanks to this atom
        for var in self.atom.pure_variables:
            if var in unsafe_variables:
                unsafe_variables.remove(var)

        # We immediately make every variable safe
        # This is fine since we modify the set without copying it
        # So, if an another node makes unsafe variables safe, the variables that are made safe now won't be make safe again
        # That is, we do not add useless atoms in the body of the rules
        s = self._add_atoms_for_unsafe_variables(for_safe_variables, unsafe_variables)
        return s + self.atom.to_ASP()

    def _get_primary_keys(self) -> List[query.Variable]:
        return list(self.atom.primary_variables)

    def __str__(self):
        return ("NOT " if self.negation else "") + str(self.atom)

class SubtreeNode(Leaf):
    def __init__(self, subtree : Tree, atom_node : AtomNode, changes_in_names : Dict[query.Variable, query.Variable], negation : bool = False):
        super().__init__(negation)
        self.subtree = subtree
        self.atom_node = atom_node
        self.changes_in_names = changes_in_names

    def apply_negation(self) -> SubtreeNode:
        return SubtreeNode(self.subtree, self.atom_node, self.changes_in_names, not self.negation)

    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        next_head_name = next_subtree_head_name(head_name[:-2])

        free_var_for_subtree = self.subtree.free_variables
        
        in_parenthesis = "(" + ", ".join(map(lambda x: x.to_ASP(), free_var_for_subtree)) + ")"

        s = self._add_atoms_for_unsafe_variables(for_safe_variables, unsafe_variables)
        s += "not " if self.negation else ""
        s += next_head_name + "_1"
        if len(free_var_for_subtree) != 0:
            s += in_parenthesis
        s += ".\n"
        s += self.subtree.to_ASP(next_head_name)
        return s

    def _get_primary_keys(self) -> List[query.Variable]:
        raise NotImplementedError("A Subtree node does not have primary keys")

    def __str__(self):
        return ("NOT " if self.negation else "") + "SUBTREE"

class EqualityNode(Leaf):
    def __init__(self, right : query.Variable, left : query.Variable, original_atom : query.Atom, negation : bool = False):
        super().__init__(negation)
        self.right = right
        self.left = left
        self.original_atom = original_atom

    def apply_negation(self) -> EqualityNode:
        return EqualityNode(self.right, self.left, self.original_atom, not self.negation)

    def to_ASP(self, head_name : str, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], unsafe_variables : Set[query.Variable]) -> str:
        s = self._add_atoms_for_unsafe_variables(for_safe_variables, unsafe_variables)

        comp = "!=" if self.negation else "="
        s += self.left.to_ASP() + comp + self.right.to_ASP() + ".\n"
        return s

    def _get_primary_keys(self) -> List[query.Variable]:
        raise NotImplementedError("An Equality node does not have primary keys")

    def __str__(self):
        return str(self.left) + ("!=" if self.negation else "=") + str(self.right)

class Tree:
    def __init__(self, q : query.Query, atom : query.Atom, free_variables : List[query.Variable], for_safe_variables : List[query.Atom], subtree : Tree = None):
        '''
        Creation of the logical tree.

        For details, see ASPpip.pdf.

        The atom can not contain a variable 'zeta'!

        :param q: the query
        :param atom: the atom to use for the rewriting
        :param free_variables: the list of free variables in the query
        :param subtree: the subformula, if the current formula needs one
        '''
        self.free_variables = free_variables
        self.for_save_variables = for_safe_variables
        primary_variables = atom.primary_variables
        secondary_variables = atom.secondary_variables

        if len(secondary_variables) == 0 and subtree is None:
            # Only primary variables and no subtree means that we have nothing in the right part of the implication
            self.root = ExistsNode([AtomNode(atom)])
            self.close_body = True
        else:
            # At least one non-primary variable
            self.close_body = False

            # We create the tree
            # First, the part that checks if the block exists
            exists_blocks = ExistsNode([AtomNode(atom)])

            # Then, we create a new atom with new, unused variables and without constants
            # This is used in the right part of the implication
            # Note that the first occurrence is not modified
            seen_names : Set[str] = set()
            different_secondary_variables : List[query.Variable] = []
            for i, var in enumerate(secondary_variables):
                in_primary = len(list(filter(lambda prim: prim.name == var.name, primary_variables))) != 0
                in_free_variables = len(list(filter(lambda free: free.name == var.name, free_variables))) != 0
                # If the variable appears in the primary keys or is a constant or is a free variable, we must always replace it in the secondary variables
                if in_primary or in_free_variables or isinstance(var, query.Constant) or var.name in seen_names:
                    new_var = query.Variable("zeta_{}".format(i))
                    different_secondary_variables.append(new_var)
                else:
                    different_secondary_variables.append(var)
                    seen_names.add(var.name)

            all_variables : List[query.Variable] = list(primary_variables) + different_secondary_variables
            atom_with_different_variables = AtomNode(query.Atom(atom.name, all_variables))

            # We memorize the changes in names we just performed
            changes_in_names : Dict[query.Variable, query.Variable] = {}
            for i, var in enumerate(secondary_variables):
                if var not in changes_in_names:
                    changes_in_names[var] = different_secondary_variables[i]

            # We create the conjunction on the equality constraints
            # Note that we do not create X = X constraints (for obvious reasons)
            # Also, we iterate in reverse order to be able to check if the variable is used multiple times (and to reduce the number of iterations needed for that check)
            and_children : List[Node] = []
            for i, var in reversed(list(enumerate(secondary_variables))):
                if var.name != different_secondary_variables[i].name:
                    # The variable is used in the primary keys
                    prim_keys_with_same_name = list(filter(lambda prim: prim.name == var.name, primary_variables))
                    if len(prim_keys_with_same_name) > 0:
                        and_children.append(EqualityNode(prim_keys_with_same_name[0], different_secondary_variables[i], atom))

                    # The variable is in the free variables
                    in_free_var = False
                    free_variables_with_same_name = list(filter(lambda free_var: free_var.name == var.name, free_variables))
                    if len(free_variables_with_same_name) > 0:
                        and_children.append(EqualityNode(free_variables_with_same_name[0], different_secondary_variables[i], atom))
                        in_free_var = True
                    
                    # The variable appears multiple times in the secondary variables
                    # AND does NOT appear in the free variables
                    # Indeed, it is useless to have ZETA_0 = X, ZETA_1 = X and ZETA_0 = ZETA_1, since ZETA_0 = X and ZETA_1 = X is enough
                    if not in_free_var:
                        for j, secondary_var in enumerate(secondary_variables[:i]):
                            if secondary_var.name == var.name:
                                if different_secondary_variables[i].name != var.name:
                                    and_children.append(EqualityNode(different_secondary_variables[i], different_secondary_variables[j], atom))
                                break
                    
                    # The variable is a constant
                    if isinstance(var, query.Constant):
                        and_children.append(EqualityNode(var, different_secondary_variables[i], atom))

            # If we have a subformula, we use it
            if subtree is not None:
                and_children.append(SubtreeNode(subtree, atom_with_different_variables, changes_in_names))

            # The implication, the for all and the AND after the first Exists
            if len(and_children) == 0:
                # No equality constraint nor subtree
                self.root = ExistsNode([exists_blocks])
                self.close_body = True
            elif len(and_children) == 1:
                implication_node = ImpliesNode([atom_with_different_variables, and_children[0]])
                not_implication_node = implication_node.apply_negation()
                # We do not need to distribute the AND (since we do not have an OR)
                not_for_all_node = ExistsNode([not_implication_node])
                big_and_node = AndNode([exists_blocks, NotNode([not_for_all_node])])
                self.root = ExistsNode([big_and_node])
            else:
                implication_node = ImpliesNode([atom_with_different_variables, AndNode(and_children)])
                not_implication_node = implication_node.apply_negation()
                distributed_or = not_implication_node.distribute_over_or()
                not_for_all_node = ExistsNode([distributed_or])
                or_node = not_for_all_node.descend_into_OR()
                not_node = NotNode([or_node])
                big_and_node = AndNode([exists_blocks, not_node])
                self.root = ExistsNode([big_and_node])
                


    def to_ASP(self, head_name : str) -> str:
        '''
        Writes the tree as an ASP program.
        
        :param head_name: The name to use for the rules of this tree. It must be of shape 'alpha_1'.
        :return: A string giving the ASP program
        '''
        if len(self.free_variables) == 0:
            s = head_name + "_1"
        else:
            in_parenthesis = "(" + ", ".join(map(lambda var: var.to_ASP(), self.free_variables)) + ")"
            s = head_name + "_1" + in_parenthesis
        s += " :- " + self.root.to_ASP(head_name + "_1", self.free_variables, self.for_save_variables, set(self.free_variables))
        if self.close_body:
            # In some cases, the tree is not as usual.
            # That is, the rightmost node is not an EqualityNode nor a SubtreeNode.
            # So, we need to manually close the body
            return s + ".\n"
        else:
            return s

    def __str__(self) -> str:
        return "TREE[" + str(self.root) + "]"

def fo_rewriting(q : query.Query, removed_atoms : List[query.Atom] = []) -> Optional[Tree]:
    '''
    Rewrites the query in FO.
    It returns a tree representing the formula in propositional logic.
    The tree can then easily be used to construct an ASP program.

    See ASPpip.pdf for details and explanations on the logic used.

    :param q: The query to rewrite
    :return: The tree describing the rewritten formula
    '''
    graph = attack_graph.AttackGraph(q)
    if not graph.is_acyclic:
        return None
    if graph.number_of_atoms > 1:
        for i, R in enumerate(graph.unattacked_atoms):
            # It may happen that we remove an atom we shouldn't
            try:
                q_reduced = q.remove_atom(R)
            except query.QueryCreationException:
                continue

            tree_for_q_reduced = fo_rewriting(q_reduced, removed_atoms=removed_atoms + [R])
            if tree_for_q_reduced is None:
                return None

            tree_for_q = Tree(q, R, q.free_variables, removed_atoms + [R], subtree=tree_for_q_reduced)
            return tree_for_q
        return None
    else:
        R = graph.unattacked_atoms[0]
        tree_for_q = Tree(q, R, q.free_variables, removed_atoms + [R])
        if tree_for_q is None:
            return None
        return tree_for_q
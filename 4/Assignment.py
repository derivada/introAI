# CSP Assignment
# Original code by Håkon Måløy
# Updated by Xavier Sánchez Díaz
# Solved by Pablo Díaz Viñambres

from ast import main
import copy
from itertools import product as prod
import time
from venv import create


class CSP:
    def __init__(self):
        # self.variables is a list of the variable names in the CSP
        self.variables = []

        # self.domains is a dictionary of domains (lists)
        self.domains = {}

        # self.constraints[i][j] is a list of legal value pairs for
        # the variable pair (i, j)
        self.constraints = {}

    def add_variable(self, name: str, domain: list):
        """Add a new variable to the CSP.

        Parameters
        ----------
        name : str
            The name of the variable to add
        domain : list
            A list of the legal values for the variable
        """
        self.variables.append(name)
        self.domains[name] = list(domain)
        self.constraints[name] = {}

    def get_all_possible_pairs(self, a: list, b: list) -> list[tuple]:
        """Get a list of all possible pairs (as tuples) of the values in
        lists 'a' and 'b', where the first component comes from list
        'a' and the second component comes from list 'b'.

        Parameters
        ----------
        a : list
            First list of values
        b : list
            Second list of values

        Returns
        -------
        list[tuple]
            List of tuples in the form (a, b)
        """
        return prod(a, b)

    def get_all_arcs(self) -> list[tuple]:
        """Get a list of all arcs/constraints that have been defined in
        the CSP.

        Returns
        -------
        list[tuple]
            A list of tuples in the form (i, j), which represent a
            constraint between variable `i` and `j`
        """
        return [(i, j) for i in self.constraints for j in self.constraints[i]]

    def get_all_neighboring_arcs(self, var: str) -> list[tuple]:
        """Get a list of all arcs/constraints going to/from variable 'var'.

        Parameters
        ----------
        var : str
            Name of the variable

        Returns
        -------
        list[tuple]
            A list of all arcs/constraints in which `var` is involved
        """
        return [(i, var) for i in self.constraints[var]]

    def add_constraint_one_way(self, i: str, j: str,
                               filter_function: callable):
        """Add a new constraint between variables 'i' and 'j'. Legal
        values are specified by supplying a function 'filter_function',
        that should return True for legal value pairs, and False for
        illegal value pairs.

        NB! This method only adds the constraint one way, from i -> j.
        You must ensure to call the function the other way around, in
        order to add the constraint the from j -> i, as all constraints
        are supposed to be two-way connections!

        Parameters
        ----------
        i : str
            Name of the first variable
        j : str
            Name of the second variable
        filter_function : callable
            A callable (function name) that needs to return a boolean.
            This will filter value pairs which pass the condition and
            keep away those that don't pass your filter.
        """
        if j not in self.constraints[i]:
            # First, get a list of all possible pairs of values
            # between variables i and j
            self.constraints[i][j] = self.get_all_possible_pairs(
                                        self.domains[i],
                                        self.domains[j])

        # Next, filter this list of value pairs through the function
        # 'filter_function', so that only the legal value pairs remain
        self.constraints[i][j] = list(filter(lambda
                                             value_pair:
                                             filter_function(*value_pair),
                                             self.constraints[i][j]))

    def add_all_different_constraint(self, var_list: list):
        """Add an Alldiff constraint between all of the variables in the
        list provided.

        Parameters
        ----------
        var_list : list
            A list of variable names
        """
        for (i, j) in self.get_all_possible_pairs(var_list, var_list):
            if i != j:
                self.add_constraint_one_way(i, j, lambda x, y: x != y)

    def backtracking_search(self):
        """This functions starts the CSP solver and returns the found
        solution.
        """
        # Make a so-called "deep copy" of the dictionary containing the
        # domains of the CSP variables. The deep copy is required to
        # ensure that any changes made to 'assignment' does not have any
        # side effects elsewhere.
        assignment = copy.deepcopy(self.domains)

        # Run AC-3 on all constraints in the CSP, to weed out all of the
        # values that are not arc-consistent to begin with
        self.inference(assignment, self.get_all_arcs())

        # Call backtrack with the partial assignment 'assignment'
        return (self.backtrack(assignment, 0, 0))

    def backtrack(self, assignment, backtrack_calls, backtrack_failures):
        """The function 'Backtrack' from the pseudocode in the
        textbook.

        The function is called recursively, with a partial assignment of
        values 'assignment'. 'assignment' is a dictionary that contains
        a list of all legal values for the variables that have *not* yet
        been decided, and a list of only a single value for the
        variables that *have* been decided.

        When all of the variables in 'assignment' have lists of length
        one, i.e. when all variables have been assigned a value, the
        function should return 'assignment'. Otherwise, the search
        should continue. When the function 'inference' is called to run
        the AC-3 algorithm, the lists of legal values in 'assignment'
        should get reduced as AC-3 discovers illegal values.

        IMPORTANT: For every iteration of the for-loop in the
        pseudocode, you need to make a deep copy of 'assignment' into a
        new variable before changing it. Every iteration of the for-loop
        should have a clean slate and not see any traces of the old
        assignments and inferences that took place in previous
        iterations of the loop.
        """
        # Check if assignment is complete
        complete = True
        for var in assignment:
            if(len(assignment[var])) is not 1:
                complete = False
                break
        
        # Return the complete assignment
        if(complete): 
            return (assignment, backtrack_calls, backtrack_failures) 

        # Select the next variable
        var = self.select_unassigned_variable(assignment)

        # Loop through all possible values in the way specified by the order heuristic
        for value in assignment[var]:
            # Save the previous assignment here so we don't carry old inferences
            old_assign = copy.deepcopy(assignment)
            # Add value to assignment
            assignment[var] = [value] 

            # For inferences, we use the Maintaining Arc Consistency (MAC) algorithm by adding neighbouring arcs to the AC-3 queue
            if(self.inference(assignment, self.get_all_neighboring_arcs(var))):
                # Call recursively to assign next variable
                (result, backtrack_calls, backtrack_failures) = self.backtrack(assignment, backtrack_calls + 1, backtrack_failures)
                # If we found a result, go up the tree
                if result is not None: return (result, backtrack_calls, backtrack_failures)
            
            # Try next value, resetting the assignment to remove inferences for last value
            assignment = old_assign
        
        # If we couldn't find a result, backtrack to previous variable. None is our failure value
        return (None, backtrack_calls, backtrack_failures + 1)
    
    def select_unassigned_variable(self, assignment):
        """The function 'Select-Unassigned-Variable' from the pseudocode
        in the textbook. Should return the name of one of the variables
        in 'assignment' that have not yet been decided, i.e. whose list
        of legal values has a length greater than one.
        """
        # Minimum Remaning Values heuristic, we choose the variable with the least choices left in the assignment
        (min_choices, min_var) = (10, None)
        for var in assignment:
            # The variable is already assigned
            if(len(assignment[var]) == 1): continue 
            # New minimum remaining values variable
            if(len(assignment[var]) < min_choices): 
                (min_choices, min_var) = (len(assignment[var]), var)
                
        return min_var

    def inference(self, assignment, queue):
        """The function 'AC-3' from the pseudocode in the textbook.
        'assignment' is the current partial assignment, that contains
        the lists of legal values for each undecided variable. 'queue'
        is the initial queue of arcs that should be visited.
        """
        # While we have elements in the queue
        while(len(queue) > 0):
            # Arc i -> j
            (i, j) = queue.pop() 
            # Revise the domain of i for assuring arc consistency
            if(self.revise(assignment, i, j)):
                # The domain is empty, there is no solution
                if(len(assignment.get(i)) == 0):
                    return False
                # The domain was revised, add all neighbour backwards arcs except j -> i 
                # to check new assignment inconsistencies
                for k in self.constraints.get(i):
                    if k == j: continue
                    queue.append((k, i))
        return True

    def revise(self, assignment, i, j):
        """The function 'Revise' from the pseudocode in the textbook.
        'assignment' is the current partial assignment, that contains
        the lists of legal values for each undecided variable. 'i' and
        'j' specifies the arc that should be visited. If a value is
        found in variable i's domain that doesn't satisfy the constraint
        between i and j, the value should be deleted from i's list of
        legal values in 'assignment'.
        """
        revised = False
        # For every value in i's domain, we check if there is a value in j's domain that satisfies the constraints
        for i_value in assignment.get(i):
            satisfied = False
            for j_value in assignment.get(j):
                if(self.constraints.get(i).get(j).count((i_value, j_value))> 0):
                    satisfied = True
                    break
            # If there is no value in j that satisfies i, we remove it i's domain and mark the arc as revised
            if not satisfied:
                assignment.get(i).remove(i_value)
                revised = True
        return revised

def create_map_coloring_csp():
    """Instantiate a CSP representing the map coloring problem from the
    textbook. This can be useful for testing your CSP solver as you
    develop your code.
    """
    csp = CSP()
    states = ['WA', 'NT', 'Q', 'NSW', 'V', 'SA', 'T']
    edges = {'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
             'NT': ['WA', 'Q'], 'NSW': ['Q', 'V']}
    colors = ['red', 'green', 'blue']
    for state in states:
        csp.add_variable(state, colors)
    for state, other_states in edges.items():
        for other_state in other_states:
            csp.add_constraint_one_way(state, other_state, lambda i, j: i != j)
            csp.add_constraint_one_way(other_state, state, lambda i, j: i != j)
    return csp


def create_sudoku_csp(filename: str) -> CSP:
    """Instantiate a CSP representing the Sudoku board found in the text
    file named 'filename' in the current directory.

    Parameters
    ----------
    filename : str
        Filename of the Sudoku board to solve

    Returns
    -------
    CSP
        A CSP instance
    """
    csp = CSP()
    board = list(map(lambda x: x.strip(), open(filename, 'r')))

    for row in range(9):
        for col in range(9):
            if board[row][col] == '0':
                csp.add_variable('%d-%d' % (row, col), list(map(str,
                                                                range(1, 10))))
            else:
                csp.add_variable('%d-%d' % (row, col), [board[row][col]])

    for row in range(9):
        csp.add_all_different_constraint(['%d-%d' % (row, col)
                                          for col in range(9)])
    for col in range(9):
        csp.add_all_different_constraint(['%d-%d' % (row, col)
                                         for row in range(9)])
    for box_row in range(3):
        for box_col in range(3):
            cells = []
            for row in range(box_row * 3, (box_row + 1) * 3):
                for col in range(box_col * 3, (box_col + 1) * 3):
                    cells.append('%d-%d' % (row, col))
            csp.add_all_different_constraint(cells)

    return csp


def print_sudoku_solution(solution):
    """Convert the representation of a Sudoku solution as returned from
    the method CSP.backtracking_search(), into a human readable
    representation.
    """
    for row in range(9):
        for col in range(9):
            print(solution['%d-%d' % (row, col)][0], end=" "),
            if col == 2 or col == 5:
                print('|', end=" "),
        print("")
        if row == 2 or row == 5:
            print('------+-------+------')


def main():
    # We run the solver for every board, also getting the time elapsed, and backtrack calls and failures
    boards = ["easy.txt", "medium.txt", "hard.txt", "veryhard.txt"]
    for board in boards:
        board_name = board.split('.')[0]
        csp = create_sudoku_csp(board)
        print('\n--- Solving board {} ---\n'.format(board_name))
        start = time.time()
        (solution, backtrack_calls, backtrack_failures) = csp.backtracking_search()
        end = time.time()
        if solution is not None:
            print('Board {} solved in {:.2f} ms, with {} backtrack calls and {} backtrack failures'
                .format(board_name, ((end - start) * 1000), backtrack_calls, backtrack_failures))
            print('\nSolution:\n')
            print_sudoku_solution(solution)
        else:
            print('Could not solve board {}\n'.format(board_name))

main()
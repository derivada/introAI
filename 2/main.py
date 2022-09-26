from Map import Map_Obj
import numpy as np
import time

# NTNU, Fall Semester 2022
# TDT4136: Introduction to Artificial Intelligence
# Assigment 2
# Author: Pablo Diaz Viñambres

class Node():
    """
    A Node class for encapsulating the coordinates, heuristic values, and child-father
    graph-like relationships between them.
    """

    def __init__(self, coords, weight):
        """Instantiate a `Node` object with its coordinates and weight of the map cell

        Parameters
        ----------
        coords : list[int]
           Map coordinates for the node
        weight: int
            The weight of the node, that is, the time is takes to step onto the map cell
        """
        self.x = coords[0]
        self.y = coords[1]
        self.weight = weight
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None
        self.childs = []
    
    def __str__(self):
        return '{{{}, {}}}'.format(self.x, self.y)
    
    # We implement comparators for correct open node list minimum extraction
    def __lt__(self, other):
        return self.f < other.f
    def __gt__(self, other):
        return self.f > other.f
    def __eq__(self, other):
        if(other==None): return False
        return self.x == other.x and self.y == other.y

def solve(task):
    """Main function that performs the pathfinding A* algorithm. Works for all tasks
    Parameters.
    ----------
    task: int
        The task ID to process
    """
    t_start = time.time()
    # Setup map and start and goal nodes
    map = Map_Obj(task)
    int_map = map.get_maps()[0]
    start_coords = map.get_start_pos()
    goal_coords = map.get_goal_pos()
    start = Node(start_coords, int_map[start_coords[0]][start_coords[1]])
    goal = Node(goal_coords, int_map[goal_coords[0]][goal_coords[1]])
    attach_parent(start, None, goal)
    open = []
    closed = [] 
    solution = None
    open.append(start)
    max_len = 1
    node_count = 0
    # Agenda loop
    while(len(open) != 0):
        # Move the goal on task 5, update the goal node and recompute heuristics
        if(task == 5):
            map.tick()
            new_goal_coords = map.get_goal_pos()
            # If goal coordinates changed, update goal position and open node heuristics
            if(new_goal_coords[0] != goal_coords[0] or new_goal_coords[1] != goal_coords[1]):
                goal_coords = new_goal_coords
                goal = Node(goal_coords, int_map[goal_coords[0]][goal_coords[1]])
                for n in open:
                    n.h = np.sqrt((goal.x - n.x)**2 + (goal.y - n.y)**2)
                    n.f = n.g + n.h

        # Update open and closed lists
        node_count += 1
        if(len(open) > max_len):
            max_len = len(open)
        parent = min(open)
        open.remove(parent)
        closed.append(parent)
        # Check for solution found
        print('Expanding {{{},{}}}, f = {}'.format(parent.x, parent.y, parent.f))
        if(is_solution(parent, goal)):
            solution = parent
            break
        # Generate succesors
        succ_list = expand(parent, int_map)
        # Iterate through succesors
        for succ in succ_list:
            # Succesor was never found
            if succ not in open and succ not in closed:
                attach_parent(succ, parent, goal)
                open.append(succ)
            # A better parent was found
            elif parent.g + int_map[succ.x, succ.y] < succ.g:
                attach_parent(succ, parent, goal)
                # A better parent was found for an already expanded child, propagate down
                # the improvement down that path
                if succ in closed:
                    propagate_path_improvements(succ)
    t_end = time.time()

    # Print solution (if found) and stats
    print('Time elapsed: {time: .3f} seconds'.format(time = t_end - t_start))
    print('Nodes explored: {}'.format(node_count))
    print('Max length of open node list: {}'.format(max_len))
    if(solution != None):
        print_path(solution, map)
    else:
        print('No solution was found!')

def expand(node, int_map):
    """Expands the current `node` to the 4 adjacent map cells

    Parameters
    ----------
    node : Node
        Node to be expanded
    int_map : int[][]
        Integer representation of the map used for finding walls and node weights
    """
    # Generates succesors
    succ_coords = [[node.x+1, node.y], [node.x-1, node.y], [node.x, node.y+1], [node.x, node.y-1]]
    succ = []
    for coords in succ_coords:
        weight = int_map[coords[0]][coords[1]]
        if weight == -1: # Wall
            continue 
        else: # Other cell type (valid for both task 1 and task 2 int maps)
            succ.append(Node(coords, weight))
    return succ

def attach_parent(child, parent, goal):
    """Defines the relationship parent-son between two nodes. Also computes the heuristic
    values for `child` from the `goal` passed and `parent` g-value.
    If `parent == None` then `g` is set to zero on the child (start position)

    Parameters
    ----------
    child : Node
    parent: Node or None
    goal : Node
    """
    child.parent = parent
    if(parent != None):
        child.g = parent.g + child.weight
        parent.childs.append(child)
    else:
        child.g = 0 # For start cell
    # We use norm 2 distance as our heuristic
    child.h = np.sqrt((goal.x - child.x)**2 + (goal.y -child.y)**2)
    child.f = child.g + child.h

def propagate_path_improvements(parent):
    """Propagates the improvement on the `parent` g-value to his descendants
    after finding a better path to him. This method is called recursively until
    there isn't any improvement on the next descendant

    Parameters
    ----------
    parent : Node
    """
    for child in parent.childs:
        if(parent.g + child.weight < child.g):
            child.parent = parent
            child.g = parent.g + child.weight
            child.f = child.g + child.h
            propagate_path_improvements(child)

def is_solution(node, goal):
    return node == goal

def print_path(node, map):
    """Prints the map in both console format as a connected list of coordinates and in
    image format by modifying the string map values. Note that a new color was added
    to the table on `Map.print_map()` for path printing.
    
    Parameters
    ----------
    node : Node
        Last node of the solution (the goal if the solution is complete)
    map: Map_Obj
        The map for the task in execution
    """
    path_stack = []
    while(node.parent != None):
        path_stack.append(node)
        node = node.parent
    print('Path length: {}'.format(len(path_stack)))
    while(len(path_stack) > 1):
        node = path_stack.pop()
        map.str_map[node.x][node.y] = ' P '
        print('{{{}, {}}} -->'.format(node.x, node.y), end = " ")
    node = path_stack.pop()
    print('{{{}, {}}}'.format(node.x, node.y))
    map.show_map()

if __name__ == '__main__':
    # solve() function solves all tasks
    # Tasks 1-4 are handled the same, 5 is a bit different
    solve(1)     
    # solve(2)     
    # solve(3)     
    # solve(4) 
    # solve(5) 
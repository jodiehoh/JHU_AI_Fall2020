# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from util import Stack

    stack = Stack()  # stack of Nodes
    visited = set()

    node = Node(problem.getStartState(), None, 0, None)
    stack.push(node)

    while not stack.isEmpty():
        curr = stack.pop()
        visited.add(curr.state)

        if problem.isGoalState(curr.state):
            return path(curr, problem)

        successors = problem.getSuccessors(curr.state)
        for s in successors:
            if s[0] not in visited:
                newNode = Node(s[0], s[1], s[2], curr)
                stack.push(newNode)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    visited = set()
    ans = list()

    parent = {problem.getStartState() : None}
    direction = {problem.getStartState() : 0}

    queue = Queue()
    queue.push((problem.getStartState(), '0'))
    visited.add(problem.getStartState())


    while not queue.isEmpty():

        node = queue.pop()

        if(problem.isGoalState(node[0])):
            curr = node[0]
            while(curr is not None):
                ans.append(direction[curr])
                curr = parent[curr]
            ans.pop()
            ans.reverse()
            return ans

        for succ in problem.getSuccessors(node[0]):
            if (succ[0] not in visited):
                queue.push(succ)
                visited.add(succ[0])
                parent[succ[0]] = node[0]
                direction[succ[0]] = succ[1]


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    visited = set()
    ans = list()

    parent = {problem.getStartState() : None}
    direction = {problem.getStartState() : 0}
    cost = {problem.getStartState() : 0}

    queue = PriorityQueue()
    queue.push((problem.getStartState(), 0, 0), 0)
    visited.add(problem.getStartState())


    while not queue.isEmpty():

        node = queue.pop()

        if(problem.isGoalState(node[0])):
            end = True
            curr = node[0]
            while(curr is not None):
                ans.append(direction[curr])
                curr = parent[curr]
            ans.pop()
            ans.reverse()
            return ans

        for succ in problem.getSuccessors(node[0]):
            if(succ[0] not in visited or cost[node[0]] + succ[2] < cost[succ[0]]):
                visited.add(succ[0])
                cost[succ[0]] = cost[node[0]] + succ[2]
                direction[succ[0]] = succ[1]
                queue.push((succ[0], succ[1], cost[succ[0]]), cost[succ[0]])
                parent[succ[0]] = node[0]



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    open = PriorityQueue()
    closed = set()
    map = dict()  # states -> node(state, direction, cost, parent)

    node = Node(problem.getStartState(), None, 0, None)
    map[problem.getStartState()] = node
    
    open.push(problem.getStartState(), 0)

    while not open.isEmpty():
        curr = open.pop()
        closed.add(curr)

        if problem.isGoalState(curr):
            return path(map[curr], problem)

        successors = problem.getSuccessors(curr)
        for s in successors:
            newCost = map[curr].cost + s[2]
            if s[0] not in closed:
                # if its in map, that means its in open, so if new cost is higher we dont want it
                if (map.get(s[0]) != None) and (newCost > map[s[0]].cost):
                    continue

                newNode = Node(s[0], s[1], newCost, map[curr])
                map[s[0]] = newNode
                f = newNode.cost + heuristic(newNode.state, problem)
                open.update(newNode.state, f)


def path(node, problem):
    actions = list()

    while node.state != problem.getStartState():
        actions.append(node.direction)
        node = node.parent

    actions.reverse()
    return actions


class Node:

    state = (-1, -1)
    direction = None
    cost = 0
    parent = None

    def __init__(self, state, direction, cost, parent):
        self.state = state
        self.direction = direction
        self.cost = cost
        self.parent = parent


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

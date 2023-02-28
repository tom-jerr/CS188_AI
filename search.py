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
import copy


import sys
import inspect
import heapq, random

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

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
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
    print("Start's childs:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    start = problem.getStartState()
    visited = []
    stack.push((start,[]))
    while not stack.isEmpty():
        node = stack.pop()
        state = node[0]
        actions = node[1]
        if problem.isGoalState(state):
            return(actions)
        for child in problem.getSuccessors(state):
            if child[0] not in visited :
                actions=copy.deepcopy(node[1])
                actions.append(child[1])
                stack.push((child[0],actions))
        visited.append(state)
    return(actions)
    util.raiseNotDefined()
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #difference also in the way to extend visited nodes
    queue = util.Queue()
    start = problem.getStartState()
    visited = []
    queue.push((start,[]))
    visited.append(start)
    while not queue.isEmpty():
        node = queue.pop()
        state = node[0]
        actions = node[1]
        if problem.isGoalState(state):
            return(actions)

        for child in problem.getSuccessors(state):
            if child[0] not in visited :
                actions=copy.deepcopy(node[1])
                actions.append(child[1])
                queue.push((child[0],actions))
                visited.append(child[0])
    return(actions)
    util.raiseNotDefined()

#change some codes in the priority queue
class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i[0] == item[0]: #changed here 
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
            else:
                self.push(item, priority)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #the item in the priorityqueue is (state,actions,priority)
    #change the codes in the priorityqueue in util file
    priorityQueue = PriorityQueue()
    start = problem.getStartState()
    visited = []
    priorityQueue.push((start,[],0),0)
    used =[]
    while not priorityQueue.isEmpty():
        node=priorityQueue.pop()
        state = node[0]
        actions=node[1]
        priority = node[2]
        visited.append(node[0])
        if problem.isGoalState(state):
            return(actions)
        for child in problem.getSuccessors(state):
            if child[0] not in visited :
                actions=copy.deepcopy(node[1])
                actions.append(child[1])
                if child[0] in used:
                    priorityQueue.update((child[0], actions, priority), priority+child[2])
                else:
                    priorityQueue.push((child[0], actions, priority+child[2]), priority + child[2])
                used.append(child[0])
            
    return(actions)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = PriorityQueue()
    start = problem.getStartState()
    visited = []
    priorityQueue.push((start, [], 0), 0 + heuristic(start, problem))
    used =[]
    while not priorityQueue.isEmpty():
        node = priorityQueue.pop()
        state = node[0]
        actions=node[1]
        priority = node[2]
        visited.append(node[0])
        if problem.isGoalState(state):
            return(actions)
        for child in problem.getSuccessors(state):
            if child[0] not in visited :
                actions = copy.deepcopy(node[1])
                actions.append(child[1])
                if child[0] in used:
                    priorityQueue.update((child[0], actions, priority), priority + child[2] + heuristic(child[0], problem))
                else:
                    priorityQueue.push((child[0], actions, priority+child[2]), priority + child[2] + heuristic(child[0] ,problem))
                used.append(child[0])
            
    return(actions)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

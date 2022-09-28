# multiAgents.py
# --------------
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


from asyncio.windows_events import INFINITE
from cmath import inf
from json.encoder import INFINITY
from multiprocessing.sharedctypes import Value
from pickle import TRUE
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print('Successor game state = ', successorGameState)
        #print('New position = ', newPos)
        #print('New food = ', newFood)
        #print('New ghost times = ', newGhostStates[0])
        #print('New scared times = ', newScaredTimes)

       
        # Baseline position value, minimum is 100 so the biases work correctly
        value = successorGameState.getScore()  

        # Find (Manhattan) distance to closest not scared ghost. Later we could use A* search
        min_ghost_dist = INFINITE
        sum_ghost_dist = 0
        for i in range(0, len(newGhostStates)):
            mh_dist = manhattanDistance(newGhostStates[i].getPosition(), newPos)
            if(newScaredTimes[i] == 0): # Not scared
                if(mh_dist < min_ghost_dist):
                    min_ghost_dist = mh_dist
                sum_ghost_dist += 2 * mh_dist
            elif(newScaredTimes[i] > mh_dist): # Scared and we could eat it
                value += 300 / mh_dist 

        #print('Dist = {}'.format(min_ghost_dist))
        if(min_ghost_dist < 1): # Loses
            return -INFINITE

        # Chase food
        min_food_dist = INFINITE
        sum_food_dist = 0
        # For now we scan the whole array
        for i in range(newFood.width):
            for j in range(newFood.height):
                if(newFood[i][j]):
                    # There is food on that square
                    mh_dist = manhattanDistance((i, j), newPos) 
                    # For food we do an inverse distance sum
                    sum_food_dist += 10 / mh_dist
                    if(mh_dist < min_food_dist):
                        min_food_dist = mh_dist

        # Add food advantage
        #print('Min food dist = {}'.format(min_food_dist))
        #print(min_food_dist)
        value += sum_food_dist
        """print('To pos: {}{}, action = {}, value = {}'.format(successorGameState.getPacmanPosition()[0],
        successorGameState.getPacmanPosition()[1], action, value))"""
        if(action is Directions.STOP): # Stopping is usually not good
            value *= 0.5

        """print('Moving from ({}, {}) to ({}, {}) VALUE = {}'.format(
            currentGameState.getPacmanPosition()[0],
            currentGameState.getPacmanPosition()[1],
            successorGameState.getPacmanPosition()[0],
            successorGameState.getPacmanPosition()[1],
            value))"""
        return value

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        (value, action) = self.max_value(gameState, 0)
        return action
    
    def max_value(self, gameState, currentDepth):
        if(gameState.isWin() or gameState.isLose() or currentDepth == self.depth):
            return (self.evaluationFunction(gameState), None)
        (bestValue, bestAction) = -INFINITY, None
        for action in gameState.getLegalActions(0):
            print('depth = {}, agent = PACMAN, move = {}'.format(currentDepth, action))
            # Call MIN for ghost #1
            (value2, action2) = self.min_value(gameState.generateSuccessor(0, action), currentDepth, 1) 
            if value2 > bestValue:
                (bestValue, bestAction) = (value2, action)
        return (bestValue, bestAction)
    
    def min_value(self, gameState, currentDepth, ghost_number):
        if(gameState.isWin() or gameState.isLose() or currentDepth == self.depth):
            return (self.evaluationFunction(gameState), None)
        (bestValue, bestAction) = -INFINITY, None
        for action in gameState.getLegalActions(ghost_number):
            # If ghost number is less than total ghosts, call MIN again for the next ghost
            for i in range(1, ghost_number):
                print('\t', end='')
            print('depth = {}, agent = GHOST #{}, move = {}'.format(currentDepth, ghost_number, action))
            if(ghost_number < gameState.getNumAgents()):
                ghost_number += 1
                (value2, action2) = self.min_value(gameState.generateSuccessor(ghost_number, action), currentDepth, ghost_number) 
            else:
                # Else, increment the depth and call MAX
                currentDepth += 1
                (value2, action2) = self.max_value(gameState.generateSuccessor(ghost_number, action), currentDepth) 
            if value2 < bestValue:
                (bestValue, bestAction) = (value2, action)
        return (bestValue, bestAction)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

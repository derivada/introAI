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
    Minimax agent (Q2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        # We start by calling the max function with 0 current depth
        (value, action) = self.max_value(gameState, 0)
        return action


    def max_value(self, game_state, cur_depth):
        """
        The max value searches on the game tree and returns the best action for Pacman based on the minmax algorithm
        It will return the current score if cur_depth reaches the maximum depth (self.depth)
        
        Parameters
        ----------
        game_state: GameState
        The current GameState, with agent and moves info
        cur_depth: int
        The depth at which we are on the game tree
        """
        # Checks if the game is on a terminal state or we have reached the maximum depth, and returns the current score
        if(game_state.isWin() or game_state.isLose() or cur_depth == self.depth):
            return (self.evaluationFunction(game_state), None)
        
        (best_val, best_action) = -INFINITY, None

        # Calls min_value on all possible actions to find the minimum value (or maximum ghost performance) for each of them
        for action in game_state.getLegalActions(0):
            (value2, action2) = self.min_value(game_state.generateSuccessor(0, action), cur_depth, 1) 
            # If we have found a better minimum value, we update it and select our next move
            if value2 > best_val:
                (best_val, best_action) = (value2, action)
        # We return the best value and the best action (the action will only be of interest at depth = 0)
        return (best_val, best_action)
    
    def min_value(self, game_state, cur_depth, ghost_number):
        """
        The min value searches on the game tree and returns the best action for a ghost based on the minmax algorithm
        It will return the current score if cur_depth reaches the maximum depth (self.depth)

        Most of the code mirrors max_value, except the mutliple ghost agent

        Parameters
        ----------
        game_state: GameState
        The current GameState, with agent and moves info
        cur_depth: int
        The depth at which we are on the game tree
        ghost_number: int
        The ghost for which we are calculating the best action
        """
        if(game_state.isWin() or game_state.isLose() or cur_depth == self.depth):
            return (self.evaluationFunction(game_state), None)
        (best_val, best_action) = INFINITY, None
        for action in game_state.getLegalActions(ghost_number):
            # If the ghost number is less than the total number of ghosts, we call min_value on the next ghost
            if(ghost_number < game_state.getNumAgents() - 1):
                (value2, action2) = self.min_value(game_state.generateSuccessor(ghost_number, action), cur_depth, ghost_number + 1) 
            else:
                # Else, we increase the depth (1 depth = 1 turn for Pacman and all the other ghosts) and call max_value
                (value2, action2) = self.max_value(game_state.generateSuccessor(ghost_number, action), cur_depth + 1) 
            if value2 < best_val:
                (best_val, best_action) = (value2, action)
        return (best_val, best_action)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning (Q3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. Uses alpha-beta pruning to speed the algorithm
        and make a deeper search possible.

        Most of the code is similar to MinimaxAgent, we only document the new pieces
        that perform the alpha-beta tree pruning.
        """ 
        # We call max_value with an alpha and beta values of -inf and inf
        # The "alpha value" is the value of the best choice we have found so far along the path for Pacman (MAX)
        # The "beta value"is the value of the best choice we have found so far along the path for the ghosts (MIN)
        (value, action) = self.max_value(gameState, 0, -INFINITE, INFINITE)
        return action
    
    def max_value(self, game_state, cur_depth, alpha, beta):
        if(game_state.isWin() or game_state.isLose() or cur_depth == self.depth):
            return (self.evaluationFunction(game_state), None)
        (best_val, best_action) = -INFINITY, None
        for action in game_state.getLegalActions(0):
            (value2, action2) = self.min_value(game_state.generateSuccessor(0, action), cur_depth, 1, alpha, beta) 
            if value2 > best_val:
                (best_val, best_action) = (value2, action)
                # We update the alpha value to be our next best choice found on the max_value path (the maximum value so far)
                alpha = max(alpha, best_val)
            # If our best value is already bigger (better for Pacman) than beta, that means the ghosts will never continue
            # down this path since Pacman is going to get a better score. We prune this subtree and return the current best_value and move
            if best_val > beta:
                return (best_val, best_action)
        return (best_val, best_action)
    
    def min_value(self, game_state, cur_depth, ghost_number, alpha, beta):
        if(game_state.isWin() or game_state.isLose() or cur_depth == self.depth):
            return (self.evaluationFunction(game_state), None)
        (best_val, best_action) = INFINITY, None
        for action in game_state.getLegalActions(ghost_number):
            # If ghost number is less than total ghosts, call MIN again for the next ghost
            if(ghost_number < game_state.getNumAgents() - 1):
                (value2, action2) = self.min_value(game_state.generateSuccessor(ghost_number, action), cur_depth, ghost_number + 1, alpha, beta) 
            else:
                # Else, increment the depth and call MAX
                (value2, action2) = self.max_value(game_state.generateSuccessor(ghost_number, action), cur_depth + 1, alpha, beta) 
            if value2 < best_val:
                (best_val, best_action) = (value2, action)
                # We update the beta value to be our next best choice found on the min_value path (the minimum value so far)
                beta = min(beta, best_val)
            # If our best value is already smaller (better for the ghosts) than alpha, that means Pacman will never continue
            # down this path since Pacman is going to get a worse score. We prune this subtree and return the current best_value and move
            if best_val < alpha:
                return (best_val, best_action)
        return (best_val, best_action)

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

#Charissa Zou (czou9)
#Jodie Hoh (jhoh5)
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        "*** YOUR CODE HERE ***"
        ghostPositions = list()
        foodPositions = list()
        closestDistanceGhost = -1
        closestDistanceFood = -1

        for ghosts in newGhostStates:
          if (ghosts.scaredTimer == 0):
            ghostPositions.append(manhattanDistance(newPos, ghosts.getPosition()))

        if (len(ghostPositions) == 0):
          closestDistanceGhost = 10000
        else:
          closestDistanceGhost = min(ghostPositions)

        for food in newFood.asList():
            foodPositions.append(manhattanDistance(newPos, food))

        if (len(foodPositions) == 0):
          closestDistanceFood = 0
        else:
          closestDistanceFood = min(foodPositions)

        score = 8 / (closestDistanceGhost + 1) + closestDistanceFood / 3
        return successorGameState.getScore() - score

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
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.depth, 0)[1]

    def minimax(self, gameState, depth, agentIndex):
      if (agentIndex == gameState.getNumAgents()):
        agentIndex = 0
        depth = depth - 1

      if (depth == 0 or gameState.isWin() or gameState.isLose()):
        return (self.evaluationFunction(gameState), Directions.STOP)

      if (agentIndex == 0):
        return self.maxAction(gameState, depth, agentIndex)
      else:
        return self.minAction(gameState, depth, agentIndex)

    def maxAction(self, gameState, depth, agentIndex):
      actions = gameState.getLegalActions(agentIndex)
      bestScore = float('-inf')
      bestAction = None
      for action in actions:
        sucessor = gameState.generateSuccessor(agentIndex, action)
        score = self.minimax(sucessor, depth, agentIndex + 1)[0]
        if (score > bestScore):
          bestScore = score 
          bestAction = action 
      return [bestScore, bestAction]

    def minAction(self, gameState, depth, agentIndex):
      actions = gameState.getLegalActions(agentIndex)
      bestScore = float('inf')
      bestAction = None
      for action in actions:
        sucessor = gameState.generateSuccessor(agentIndex, action)
        score = self.minimax(sucessor, depth, agentIndex + 1)[0]
        if (score < bestScore):
          bestScore = score 
          bestAction = action 
      return [bestScore, bestAction]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.minimax(gameState, self.depth, 0, float('-inf'), float('inf'))[1]

    def minimax(self, gameState, depth, agentIndex, alpha, beta):
      if (agentIndex == gameState.getNumAgents()):
        agentIndex = 0
        depth = depth - 1

      if (depth == 0 or gameState.isWin() or gameState.isLose()):
        return (self.evaluationFunction(gameState), Directions.STOP)

      if (agentIndex == 0):
        return self.maxAction(gameState, depth, agentIndex, alpha, beta)
      else:
        return self.minAction(gameState, depth, agentIndex, alpha, beta)

    def maxAction(self, gameState, depth, agentIndex, alpha, beta):
      actions = gameState.getLegalActions(agentIndex)
      bestScore = float('-inf')
      bestAction = None
      for action in actions:
        sucessor = gameState.generateSuccessor(agentIndex, action)
        score = self.minimax(sucessor, depth, agentIndex + 1, alpha, beta)[0]
        if (score > bestScore):
          bestScore = score 
          bestAction = action 
        if (bestScore > beta):
          return [bestScore, bestAction]
        alpha = max({alpha, score})

      return [bestScore, bestAction]

    def minAction(self, gameState, depth, agentIndex, alpha, beta):
      actions = gameState.getLegalActions(agentIndex)
      bestScore = float('inf')
      bestAction = None
      for action in actions:
        sucessor = gameState.generateSuccessor(agentIndex, action)
        score = self.minimax(sucessor, depth, agentIndex + 1, alpha, beta)[0]
        if (score < bestScore):
          bestScore = score 
          bestAction = action 
        if (bestScore < alpha):
          return [bestScore, bestAction]
        beta = min({beta, score})

      return [bestScore, bestAction]

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
        return self.expectimax(gameState, self.depth, 0)[1]

    def expectimax(self, gameState, depth, agentIndex):
      if (agentIndex == gameState.getNumAgents()):
        agentIndex = 0
        depth = depth - 1

      if (depth == 0 or gameState.isWin() or gameState.isLose()):
        return (self.evaluationFunction(gameState), Directions.STOP)

      if (agentIndex == 0):
        return self.maxAction(gameState, depth, agentIndex)
      else:
        return self.average(gameState, depth, agentIndex)

    def average(self, gameState, depth, agentIndex):
      actions = gameState.getLegalActions(agentIndex)
      bestScore = float('-inf')
      bestAction = None

      totalScore = 0
      for action in actions:
        sucessor = gameState.generateSuccessor(agentIndex, action)
        score = self.expectimax(sucessor, depth, agentIndex + 1)[0]
        totalScore += score

      return (totalScore/len(actions), Directions.STOP)

    def maxAction(self, gameState, depth, agentIndex):
      actions = gameState.getLegalActions(agentIndex)
      bestScore = float('-inf')
      bestAction = None
      for action in actions:
        sucessor = gameState.generateSuccessor(agentIndex, action)
        score = self.expectimax(sucessor, depth, agentIndex + 1)[0]
        if (score > bestScore):
          bestScore = score 
          bestAction = action 
      return [bestScore, bestAction]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghostPositions = list()
    foodPositions = list()
    closestDistanceGhost = -1
    closestDistanceFood = -1

    for ghosts in newGhostStates:
      if (ghosts.scaredTimer == 0):
        ghostPositions.append(manhattanDistance(newPos, ghosts.getPosition()))

    if (len(ghostPositions) == 0):
      closestDistanceGhost = 10000
    else:
      closestDistanceGhost = min(ghostPositions)

    for food in newFood.asList():
        foodPositions.append(manhattanDistance(newPos, food))

    if (len(foodPositions) == 0):
      closestDistanceFood = 0
    else:
      closestDistanceFood = min(foodPositions)

    score = 8 / (closestDistanceGhost + 1) + closestDistanceFood / 3
    return successorGameState.getScore() - score

# Abbreviation
better = betterEvaluationFunction

def manhattanHeuristic(pos1, pos2):
    "The Manhattan distance"
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


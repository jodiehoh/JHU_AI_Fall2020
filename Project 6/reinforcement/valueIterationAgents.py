# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        it = 0
        while it < self.iterations:
            values = self.values.copy()
            for state in states:
                self.values[state] = -float('inf')
                possibleAction = self.mdp.getPossibleActions(state)

                for action in possibleAction:
                    qValue = 0
                    for transitionState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        qValue += probability * (self.mdp.getReward(state, action, transitionState) + self.discount * values[transitionState])
                    self.values[state] = max(self.values[state], qValue)
                if self.values[state] == -float('inf'):
                    self.values[state] = 0.0
            it += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        statesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)

        qValue = 0.0

        for pair in statesAndProbs:
            stateNew = pair[0]
            probability = float(pair[1])
            reward = self.mdp.getReward(state,action,stateNew)

            qValue += probability * (reward + self.discount * self.values[stateNew])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        else:
            actions = self.mdp.getPossibleActions(state)
            bestAction = None
            qValueBest = float("-inf")

            for action in actions:
                qValue = self.computeQValueFromValues(state, action)
                if (qValue >= qValueBest):
                    bestAction = action
                    qValueBest = qValue
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        it = 0
        states = self.mdp.getStates()
        length = len(states)
        new = util.Counter()

        while (it < self.iterations):
        	state = states[it % length]
        	if not self.mdp.isTerminal(state):
        		action = self.getAction(state)
        		if action is not None:
        			new[state] = self.getQValue(state, action)
        		self.values = new
        	it += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def difference(self, state):
        """
            Returns the absolute value of the difference between the current value of state
            and the highest Q-Value across all possible actions from state
        """
        maxQValue = float('-inf')

        for action in self.mdp.getPossibleActions(state):
            QValue = self.computeQValueFromValues(state, action)
            maxQValue = max(maxQValue, QValue)

        return abs(maxQValue - self.values[state])

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = collections.defaultdict(set)

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

        queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                diff = self.difference(state)
                queue.push(state, -diff)

        for _ in range(self.iterations):
            if queue.isEmpty():
                break

            state = queue.pop()

            maxQValue = float('-inf')
            for action in self.mdp.getPossibleActions(state):
                QValue = self.computeQValueFromValues(state, action)
                maxQValue = max(maxQValue, QValue)

            self.values[state] = maxQValue

            for pred in predecessors[state]:
                diff = self.difference(pred)
                if diff > self.theta:
                    queue.update(pred, -diff)

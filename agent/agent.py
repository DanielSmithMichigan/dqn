from collections import deque
from .network import Network
from .buffer import Buffer
from . import constants
from . import util
import numpy as np
import tensorflow as tf
import random
import matplotlib
import matplotlib.pyplot as plt
import time
plt.ion()

def toList(d):
    output = []
    for i in range(len(d)):
        output.append(d[i])
    return output

#TODO: Update learned network periodically + tests
#TODO: Tests on execution function
#TODO: Re-read DQN paper

class Agent:
    def __init__(
            self,
            sess,
            env,
            episodeLimit,
            maxMemoryLength,
            gamma,
            batchSize,
            numAvailableActions,
            numObservations,
            networkSize,
            learningRate,
            episodeStepLimit,
            episodeSampleSize,
            nStepUpdate,
            priorityExponent,
            minFramesForTraining,
            minFramesForTargetUpdate,
            includeIntermediatePairs,
            updateTargetNetworkPeriod,
            trainNetworkPeriod,
            numAtoms,
            valueMin,
            valueMax,
            render,
            showGraph,
            epsilonInitial,
            epsilonDecay,
            testPeriod
        ):
        self.sess = sess
        self.env = env
        self.epsilonDecay = epsilonDecay
        self.epsilon = epsilonInitial
        self.networkSize = networkSize
        self.episodeLimit = episodeLimit
        self.numAvailableActions = numAvailableActions
        self.updateTargetNetworkPeriod = updateTargetNetworkPeriod
        self.trainNetworkPeriod = trainNetworkPeriod
        self.gamma = gamma
        self.episodeStepLimit = episodeStepLimit
        self.totalEpisodeReward = 0
        self.numAtoms = numAtoms
        self.render = render
        self.showGraph = showGraph
        self.memoryBuffer = Buffer(
            maxMemoryLength=maxMemoryLength,
            nStepUpdate=nStepUpdate,
            priorityExponent=priorityExponent,
            batchSize=batchSize,
            gamma=gamma,
            includeIntermediatePairs=includeIntermediatePairs,
            sess=sess
        )
        self.learnedNetwork = Network(
            name="learned-network",
            sess=sess,
            numObservations=numObservations,
            networkSize=networkSize,
            numAvailableActions=numAvailableActions,
            learningRate=learningRate,
            numAtoms=numAtoms,
            valueMin=valueMin,
            valueMax=valueMax
        )
        self.minFramesForTraining = minFramesForTraining
        self.minFramesForTargetUpdate = minFramesForTargetUpdate
        self.support = self.sess.run(self.learnedNetwork.support)
        self.barWidth = self.support[1] - self.support[0]
        self.testPeriod = testPeriod
        self.buildGraphs()
    def getAgentAssessment(self, state):
        probabilities, maxQ = self.sess.run([
            self.learnedNetwork.probabilities,
            self.learnedNetwork.maxQ
        ], feed_dict={
            self.learnedNetwork.environmentInput: [state]
        })
        self.agentAssessmentsOverTime.append(maxQ[0])
        self.recentAgentProbabilities = probabilities[0]
    def getAction(self, state):
        action = self.sess.run(self.learnedNetwork.chosenAction, {
            self.learnedNetwork.environmentInput: [state]
        })[0]
        # self.chosenActions[action] = self.chosenActions[action] + 1
        return action
    def goToNextState(self, currentState, actionChosen, stepNum):
        nextState, reward, done, info = self.env.step(actionChosen)
        # if (stepNum > self.episodeStepLimit):
        #     reward = -100
        #     done = True
        self.totalEpisodeReward = self.totalEpisodeReward + reward
        memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
        memoryEntry[constants.STATE] = currentState
        memoryEntry[constants.ACTION] = actionChosen
        memoryEntry[constants.REWARD] = reward
        memoryEntry[constants.NEXT_STATE] = nextState
        memoryEntry[constants.GAMMA] = self.gamma if not done else 0
        memoryEntry[constants.IS_TERMINAL] = done
        memoryEntry[constants.LOSS] = 1.0
        memoryEntry[constants.PRIORITY_CACHE] = 1.0
        self.episodeMemories.append(memoryEntry)
        return nextState, done
    def train(self):
        trainingEpisodes = self.memoryBuffer.getSampleBatch()
        targets, predictions, actions = self.learnedNetwork.trainAgainst(trainingEpisodes, self.support)
        choice = np.random.randint(len(targets))
        self.recentTarget = targets[choice]
        self.recentPrediction = predictions[choice]
        self.recentAction = actions[choice]
        self.memoryBuffer.updateLossPriorityCache(trainingEpisodes)
    def buildGraphs(self):
        self.targetExample = np.zeros(self.numAtoms)
        self.actualExample = np.zeros(self.numAtoms)
        self.actionExample = 0
        self.overview = plt.figure()
        self.rewardsReceivedOverTime = []
        self.rewardsReceivedOverTimeGraph = self.overview.add_subplot(3, 2, 1)
        self.rewardsReceivedOverTimeGraph.set_ylabel('Reward')
        self.rewardsReceivedOverTimeGraph.set_xlabel('Episode #')

        self.lossesGraph = self.overview.add_subplot(3, 2, 2)
        self.lossesGraph.set_ylabel('Loss amount')
        self.lossesGraph.set_xlabel('Iteration')

        self.agentAssessmentsOverTime = []
        self.agentAssessmentGraph = self.overview.add_subplot(3, 2, 3)
        self.agentAssessmentGraph.set_ylabel('Expected Value')
        self.agentAssessmentGraph.set_xlabel('Episode #')

        self.epsilonOverTime = []
        self.epsilonOverTimeGraph = self.overview.add_subplot(3, 2, 4)
        self.epsilonOverTimeGraph.set_ylabel('Epsilon')
        self.epsilonOverTimeGraph.set_xlabel('Episode #')

        self.choicesOverTime = []
        self.choicesOverTimeGraph = self.overview.add_subplot(3, 1, 3)

        self.probabilityDistributions = plt.figure()

        self.recentTarget = np.zeros(self.numAtoms)
        self.recentTargetGraph = self.probabilityDistributions.add_subplot(6, 1, 1)
        self.recentTargetGraph.set_ylabel('Probability')
        self.recentTargetGraph.set_title("Prediction")

        self.recentPrediction = np.zeros(self.numAtoms)
        self.recentPredictionGraph = self.probabilityDistributions.add_subplot(6, 1, 2)
        self.recentPredictionGraph.set_ylabel('Probability')
        self.recentPredictionGraph.set_xlabel('Value')
        self.recentPredictionGraph.set_title("Prediction")

        self.agentAssessmentGraphs = []
        self.recentAgentProbabilities = np.zeros((self.numAvailableActions, self.numAtoms))
        for i in range(self.numAvailableActions):
            self.agentAssessmentGraphs.append(self.probabilityDistributions.add_subplot(6, 1, 3 + i))

        # self.denseWeights = plt.figure()
        # self.denseWeightSubplots = []
        # numWeightsGraphs = len(self.networkSize) + 1
        # self.weightVariables = []
        # with tf.variable_scope("learned-network", reuse=True):
        #     for i in range(len(self.networkSize)):
        #         self.denseWeightSubplots.append(self.denseWeights.add_subplot(numWeightsGraphs, 1, i + 1))
        #         self.weightVariables.append(tf.get_variable("hidden_"+str(i)+"/kernel"))
        #     self.denseWeightSubplots.append(self.denseWeights.add_subplot(numWeightsGraphs, 1, len(self.networkSize) + 1))
        #     self.weightVariables.append(tf.get_variable("logits/kernel"))

        self.recentAction = 0
    def updateGraphs(self):
        self.rewardsReceivedOverTimeGraph.cla()
        self.rewardsReceivedOverTimeGraph.plot(self.rewardsReceivedOverTime)
        self.lossesGraph.cla()
        self.lossesGraph.plot(self.learnedNetwork.losses)
        self.agentAssessmentGraph.cla()
        self.agentAssessmentGraph.plot(self.agentAssessmentsOverTime)
        self.epsilonOverTimeGraph.cla()
        self.epsilonOverTimeGraph.plot(self.epsilonOverTime)
        self.choicesOverTimeGraph.cla()
        self.choicesOverTimeGraph.plot('x', 'y1', util.getColumn(self.choicesOverTime, 0), label=constants.ACTION_NAMES[0])
        self.choicesOverTimeGraph.plot('x', 'y2', util.getColumn(self.choicesOverTime, 1), label=constants.ACTION_NAMES[1])
        self.choicesOverTimeGraph.plot('x', 'y3', util.getColumn(self.choicesOverTime, 2), label=constants.ACTION_NAMES[2])
        self.choicesOverTimeGraph.plot('x', 'y4', util.getColumn(self.choicesOverTime, 3), label=constants.ACTION_NAMES[3])
        self.choicesOverTimeGraph.legend(loc=2)
        self.overview.canvas.draw()

        self.recentTargetGraph.cla()
        self.recentTargetGraph.bar(self.support, self.recentTarget, width=self.barWidth)
        self.recentTargetGraph.set_ylabel("Target")
        self.recentTargetGraph.set_ylim(0, 1)
        self.recentTargetGraph.set_title(constants.ACTION_NAMES[self.recentAction])
        self.recentPredictionGraph.cla()
        self.recentPredictionGraph.bar(self.support, self.recentPrediction, width=self.barWidth)
        self.recentPredictionGraph.set_ylabel("Prediction")
        self.recentPredictionGraph.set_ylim(0, 1)
        for i in range(self.numAvailableActions):
            self.agentAssessmentGraphs[i].cla()
            self.agentAssessmentGraphs[i].bar(self.support, self.recentAgentProbabilities[i], width=self.barWidth)
            self.agentAssessmentGraphs[i].set_title(constants.ACTION_NAMES[i])
            self.agentAssessmentGraphs[i].set_ylim(0, 1)
        self.probabilityDistributions.canvas.draw()

        # for i in range(len(self.weightVariables)):
        #     self.denseWeightSubplots[i].cla()
        #     self.denseWeightSubplots[i].imshow(self.sess.run(self.weightVariables[i]))
        # self.denseWeights.canvas.draw()

        plt.pause(0.00001)
    def playEpisode(self):
        self.epsilon = self.epsilon * self.epsilonDecay
        self.epsilonOverTime.append(self.epsilon)
        epsilon = min(np.random.random() * self.epsilon, 1)
        state = self.env.reset()
        self.getAgentAssessment(state)
        self.totalEpisodeReward = 0
        stepNum = 0
        self.episodeMemories = []
        agentChoices = np.zeros(self.numAvailableActions)
        while True:
            stepNum = stepNum + 1
            if np.random.random() > epsilon:
                actionChosen = self.getAction(state)
            else:
                actionChosen = np.random.randint(self.numAvailableActions)
            agentChoices[actionChosen] = agentChoices[actionChosen] + 1
            state, done = self.goToNextState(state, actionChosen, stepNum)
            if self.render:
                self.env.render()
            if stepNum > self.episodeStepLimit:
                break
            if done:
                break
        self.rewardsReceivedOverTime.append(self.totalEpisodeReward)
        sumChoices = np.sum(agentChoices)
        agentChoices = agentChoices / sumChoices
        self.choicesOverTime.append(agentChoices)
        cumulativeReward = 0
        for i in reversed(self.episodeMemories):
            cumulativeReward = cumulativeReward + i[constants.REWARD]
            i[constants.REWARD] = cumulativeReward
            self.memoryBuffer.addMemory(i)
    def execute(self):
        self.sess.run(tf.global_variables_initializer())
        frames = 0
        episodeNum = 0
        for episodeNum in range(self.episodeLimit):
            self.playEpisode()
            if len(self.memoryBuffer.memory) > self.minFramesForTraining:
                self.train()
            if self.showGraph:
                self.updateGraphs()
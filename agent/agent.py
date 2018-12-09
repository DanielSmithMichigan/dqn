from collections import deque
from .network import Network
from .buffer import Buffer
from . import constants
from . import util
from .prioritized_experience_replay import PrioritizedExperienceReplay
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
            maxMemoryLength,
            gamma,
            batchSize,
            numAvailableActions,
            numObservations,
            networkSize,
            learningRate,
            episodeStepLimit,
            nStepUpdate,
            priorityExponent,
            minFramesForTraining,
            includeIntermediatePairs,
            render,
            showGraph,
            epsilonInitial,
            epsilonDecay,
            noisyLayers,
            numTestPeriods,
            numTestsPerTestPeriod,
            episodesPerTest,
            intermediateTests,
            rewardsMovingAverageSampleLength,
            maxGradientNorm,
            minExploration,
            maxExploration,
            maxRunningMinutes
        ):
        self.startTime = time.time()
        self.sess = sess
        self.env = env
        self.epsilonDecay = epsilonDecay
        self.epsilon = epsilonInitial
        self.minExploration = minExploration
        self.maxExploration = maxExploration
        self.networkSize = networkSize
        self.episodesPerTest = episodesPerTest
        self.numAvailableActions = numAvailableActions
        self.maxRunningMinutes = maxRunningMinutes
        self.gamma = gamma
        self.episodeStepLimit = episodeStepLimit
        self.totalEpisodeReward = 0
        self.render = render
        self.numTestPeriods = numTestPeriods
        self.numTestsPerTestPeriod = numTestsPerTestPeriod
        self.showGraph = showGraph
        self.numTestsPerTestPeriod = numTestsPerTestPeriod
        self.memoryBuffer = PrioritizedExperienceReplay(
            numMemories=maxMemoryLength,
            priorityExponent=priorityExponent,
            batchSize=batchSize
        )
        self.learnedNetwork = Network(
            name="learned-network-"+str(np.random.randint(100000, 999999)),
            sess=sess,
            numObservations=numObservations,
            networkSize=networkSize,
            numAvailableActions=numAvailableActions,
            learningRate=learningRate,
            noisyLayers=noisyLayers,
            maxGradientNorm=maxGradientNorm,
            batchSize=batchSize
        )
        self.minFramesForTraining = minFramesForTraining
        self.intermediateTests = intermediateTests
        self.testOutput = []
        self.rewardsMovingAverageSampleLength = rewardsMovingAverageSampleLength
        if showGraph:
            self.buildGraphs()
        #Build placeholder graph values
        self.targetExample = 0
        self.actualExample = 0
        self.actionExample = 0
        self.recentTarget = 0
        self.recentPrediction = 0
        self.rewardsMovingAverage = []
        self.rewardsReceivedOverTime = []
        self.rewardsStdDev = []
        self.agentAssessmentsOverTime = []
        self.epsilonOverTime = []
        self.choicesOverTime = []

    def getAgentAssessment(self, state):
        qValues, maxQ = self.sess.run([
            self.learnedNetwork.qValues,
            self.learnedNetwork.maxQ
        ], feed_dict={
            self.learnedNetwork.environmentInput: [state]
        })
        self.agentAssessmentsOverTime.append(maxQ[0])
        self.recentAgentQValues = qValues[0]
    def getAction(self, state):
        action = self.sess.run(self.learnedNetwork.chosenAction, {
            self.learnedNetwork.environmentInput: [state]
        })[0]
        # self.chosenActions[action] = self.chosenActions[action] + 1
        return action
    def goToNextState(self, currentState, actionChosen, stepNum):
        nextState, reward, done, info = self.env.step(actionChosen)
        self.totalEpisodeReward = self.totalEpisodeReward + reward
        memoryEntry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
        memoryEntry[constants.STATE] = currentState
        memoryEntry[constants.ACTION] = actionChosen
        memoryEntry[constants.REWARD] = reward
        memoryEntry[constants.NEXT_STATE] = nextState
        memoryEntry[constants.GAMMA] = self.gamma if not done else 0
        memoryEntry[constants.IS_TERMINAL] = done
        self.episodeMemories.append(memoryEntry)
        return nextState, done
    def train(self):
        trainingEpisodes = self.memoryBuffer.getMemoryBatch()
        targets, predictions, actions = self.learnedNetwork.trainAgainst(trainingEpisodes)
        choice = np.random.randint(len(targets))
        self.recentTarget = targets[choice]
        self.recentPrediction = predictions[choice]
        self.recentAction = actions[choice]
        self.memoryBuffer.updateMemories(trainingEpisodes)
    def buildGraphs(self):
        self.overview = plt.figure()
        self.lastNRewardsGraph = self.overview.add_subplot(4, 1, 1)

        self.rewardsReceivedOverTimeGraph = self.overview.add_subplot(4, 2, 3)
        self.rewardsReceivedOverTimeGraph.set_ylabel('Reward')
        self.rewardsReceivedOverTimeGraph.set_xlabel('Episode #')

        self.lossesGraph = self.overview.add_subplot(4, 2, 4)
        self.lossesGraph.set_ylabel('Loss amount')
        self.lossesGraph.set_xlabel('Iteration')

        self.agentAssessmentGraph = self.overview.add_subplot(4, 2, 5)
        self.agentAssessmentGraph.set_ylabel('Expected Value')
        self.agentAssessmentGraph.set_xlabel('Episode #')

        self.epsilonOverTimeGraph = self.overview.add_subplot(4, 2, 6)
        self.epsilonOverTimeGraph.set_ylabel('Epsilon')
        self.epsilonOverTimeGraph.set_xlabel('Episode #')

        self.choicesOverTimeGraph = self.overview.add_subplot(4, 1, 4)

        self.qValuesFigure = plt.figure()

        self.recentTrainingExampleGraph = self.qValuesFigure.add_subplot(2, 1, 1)

        self.qValueExample = self.qValuesFigure.add_subplot(2, 1, 2)
        self.qValueExample.set_ylabel('Probability')
        self.qValueExample.set_xlabel('Value')
        self.qValueExample.set_title("Prediction")

        self.recentAction = 0
    def updateGraphs(self):
        self.lastNRewardsGraph.cla()
        self.lastNRewardsGraph.plot(self.rewardsReceivedOverTime[-self.numTestsPerTestPeriod:], label="Reward")
        self.lastNRewardsGraph.plot(self.rewardsMovingAverage[-self.numTestsPerTestPeriod:], label="Moving average 20")
        self.rewardsReceivedOverTimeGraph.cla()
        self.rewardsReceivedOverTimeGraph.plot(self.rewardsReceivedOverTime)
        self.lossesGraph.cla()
        self.lossesGraph.plot(self.learnedNetwork.losses)
        self.agentAssessmentGraph.cla()
        self.agentAssessmentGraph.plot(self.agentAssessmentsOverTime)
        self.epsilonOverTimeGraph.cla()
        self.epsilonOverTimeGraph.plot(self.epsilonOverTime)
        self.choicesOverTimeGraph.cla()
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 0), label=constants.ACTION_NAMES[0])
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 1), label=constants.ACTION_NAMES[1])
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 2), label=constants.ACTION_NAMES[2])
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 3), label=constants.ACTION_NAMES[3])
        self.choicesOverTimeGraph.plot(util.getColumn(self.choicesOverTime, 4), label="Epsilon", linestyle=":")
        self.choicesOverTimeGraph.legend(loc=2)
        self.overview.canvas.draw()

        self.recentTrainingExampleGraph.cla()
        self.recentTrainingExampleGraph.bar(["Target", "Prediction"], [self.recentTarget, self.recentPrediction])
        self.recentTrainingExampleGraph.set_ylabel("Q Value")
        self.recentTrainingExampleGraph.set_title(constants.ACTION_NAMES[self.recentAction])
        self.qValueExample.cla()
        self.qValueExample.bar(constants.ACTION_NAMES, self.recentAgentQValues)
        self.qValueExample.set_ylabel("Q Value")
        self.qValuesFigure.canvas.draw()

        plt.pause(0.00001)
    def playEpisode(self, useRandomActions, recordTestResult, testNum=0):
        epsilon = 0
        if useRandomActions:
            self.epsilon = max(self.epsilon * self.epsilonDecay, self.minExploration)
            epsilon = min(np.random.random() * self.epsilon, 1)
            epsilon = min(epsilon, self.maxExploration)
        self.epsilonOverTime.append(self.epsilon)
        state = self.env.reset()
        self.getAgentAssessment(state)
        self.totalEpisodeReward = 0
        stepNum = 0
        self.episodeMemories = []
        agentChoices = np.zeros(self.numAvailableActions + 1)
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
        periodResults = self.rewardsReceivedOverTime[-self.numTestsPerTestPeriod:]
        mu = np.mean(periodResults)
        std = np.std(periodResults)
        self.rewardsMovingAverage.append(mu)
        self.rewardsStdDev.append(std)
        sumChoices = np.sum(agentChoices)
        agentChoices = agentChoices / sumChoices
        agentChoices[self.numAvailableActions] = epsilon
        self.choicesOverTime.append(agentChoices)
        cumulativeReward = 0
        for i in reversed(self.episodeMemories):
            cumulativeReward = cumulativeReward + i[constants.REWARD]
            i[constants.REWARD] = cumulativeReward
            self.memoryBuffer.add(i)
        if (recordTestResult):
            self.testResults.append(self.totalEpisodeReward)
    def executeTestPeriod(self):
        self.testResults = []
        for test_num in range(self.numTestsPerTestPeriod):
            self.playEpisode(useRandomActions=False,recordTestResult=True,testNum=test_num)
        print("Test "+str(len(self.testResults))+": "+str(np.mean(self.testResults)))
        self.testOutput.append(np.mean(self.testResults))
    def outOfTime(self):
        return time.time() > self.startTime + (self.maxRunningMinutes * 60)
    def execute(self):
        self.sess.run(tf.global_variables_initializer())
        for testNum in range(self.numTestPeriods):
            for episodeNum in range(self.episodesPerTest):
                self.playEpisode(useRandomActions=True,recordTestResult=False)
                if self.memoryBuffer.length > self.minFramesForTraining:
                    self.train()
                if self.outOfTime():
                    break
            if self.showGraph:
                self.updateGraphs()
            if self.intermediateTests:
                self.executeTestPeriod()
            if self.outOfTime():
                break
        self.executeTestPeriod()
        return self.testOutput
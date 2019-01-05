from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym

env = gym.make('LunarLander-v2')

sess = tf.Session()
a = Agent(
    sess=sess,
    env=env,
    numAvailableActions=4,
    numObservations=8,
    rewardsMovingAverageSampleLength=200,
    gamma=1,
    nStepUpdate=1,
    includeIntermediatePairs=False,

    # test parameters
    episodesPerTest=10000,
    numTestPeriods=100,
    numTestsPerTestPeriod=20,
    maxRunningMinutes=600,
    episodeStepLimit=1024,
    intermediateTests=False,

    render=False,
    showGraph=False,
    saveModel=True,

    # hyperparameters
    maxMemoryLength=int(1e6),
    batchSize=256,
    learningRate=1e-2,
    priorityExponent= 0,
    epsilonInitial = 2,
    epsilonDecay = .997,
    minExploration = .01,
    maxExploration = .85,
    minFramesForTraining = 2048,
    maxGradientNorm = 5,
    preNetworkSize = [256, 256],
    postNetworkSize = [512],
    numQuantiles = 24,
    embeddingDimension = 48,
    kappa = 1.0,
    trainingIterations = 3
)
testResults = a.execute()








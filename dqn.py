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
    gamma=.99,
    nStepUpdate=1,
    includeIntermediatePairs=False,

    # test parameters
    episodesPerTest=25,
    numTestPeriods=10000,
    numTestsPerTestPeriod=20,
    maxRunningMinutes=600,
    episodeStepLimit=1024,
    intermediateTests=False,

    render=False,
    showGraph=True,
    saveModel=True,
    loadModel=False,
    disableRandomActions=False,
    disableTraining=False,
    # agentName="agent_223120799",

    # hyperparameters
    rewardScaling=pow(10, -.75),
    nStepReturns=1,
    maxMemoryLength=int(1e6),
    batchSize=64,
    learningRate=6.25e-4,
    priorityExponent= 0,
    epsilonInitial = 1,
    epsilonDecay = .999,
    minExploration = .01,
    maxExploration = 1.0,
    minFramesForTraining = 2048,
    maxGradientNorm = 5,
    preNetworkSize = [256,256],
    postNetworkSize = [512],
    numQuantiles = 32,
    embeddingDimension = 64,
    kappa = 1.0,
    trainingIterations = 3,
    tau = 0.001
)
testResults = a.execute()








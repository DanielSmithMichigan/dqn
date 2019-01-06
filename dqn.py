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
    # agentName="agent_281576132",

    # hyperparameters
    maxMemoryLength=int(1e6),
    batchSize=256,
    learningRate=1e-2,
    priorityExponent= 0,
    epsilonInitial = 1,
    epsilonDecay = .9975,
    minExploration = .01,
    maxExploration = 1.0,
    minFramesForTraining = 2048,
    maxGradientNorm = 5,
    preNetworkSize = [128, 128],
    postNetworkSize = [512],
    numQuantiles = 32,
    embeddingDimension = 64,
    kappa = 1.0,
    trainingIterations = 4,
    tau = 0.01
)
testResults = a.execute()








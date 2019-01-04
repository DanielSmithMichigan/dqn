from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym

env = gym.make('LunarLander-v2')

allResults = []

for i in range(1):
    print("BEGINNING")
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
        numTestsPerTestPeriod=0,
        maxRunningMinutes=360,
        episodeStepLimit=1024,
        intermediateTests=False,

        render=False,
        showGraph=True,

        # hyperparameters
        maxMemoryLength=int(1e6),
        batchSize=256,
        learningRate=0.0006,
        priorityExponent= 0,
        epsilonInitial = 1.5,
        epsilonDecay = .998,
        minExploration = .01,
        maxExploration = .85,
        minFramesForTraining = 2048,
        maxGradientNorm = 5,
        preNetworkSize = [256],
        postNetworkSize = [256],
        numQuantiles = 16,
        embeddingDimension = 32,
        kappa = 1.0,
        trainingIterations = 3
    )
    testResults = a.execute()
    if len(allResults) > 0:
        allResults = np.vstack((allResults, testResults))
    else:
        allResults = np.array([testResults], dtype=object,ndmin=2)
    np.savetxt("./test-results/proper-gradient-clipping.txt",allResults,delimiter=",")








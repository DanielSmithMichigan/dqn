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
        rewardsMovingAverageSampleLength=20,
        gamma=1,
        nStepUpdate=1,
        includeIntermediatePairs=False,

        # test parameters
        episodesPerTest=100,
        numTestPeriods=100,
        numTestsPerTestPeriod=15,
        maxRunningMinutes=90,
        episodeStepLimit=1024,
        intermediateTests=True,

        render=False,
        showGraph=True,

        # hyperparameters
        maxMemoryLength=int(10e6),
        batchSize=256,
        networkSize=[128, 128, 512],
        learningRate=7e-4,
        priorityExponent= 0,
        epsilonInitial = 2,
        epsilonDecay = .9995,
        minExploration = .05,
        maxExploration = .85,
        minFramesForTraining = 2048,
        maxGradientNorm = 5,
        noisyLayers = False
    )
    testResults = a.execute()
    if len(allResults) > 0:
        allResults = np.vstack((allResults, testResults))
    else:
        allResults = np.array([testResults], dtype=object,ndmin=2)
    np.savetxt("./test-results/proper-gradient-clipping.txt",allResults,delimiter=",")








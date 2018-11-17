from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym

env = gym.make('LunarLander-v2')

allResults = []

for i in range(20):
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
        numTestPeriods=20,
        numTestsPerTestPeriod=20,
        episodeStepLimit=1024,
        intermediateTests=True,

        render=False,
        showGraph=False,

        # hyperparameters
        valueMin=-300.0,
        valueMax=200.0,
        numAtoms=10,
        maxMemoryLength=100000,
        batchSize=256,
        networkSize=[128,128,128,512],
        learningRate=2e-4,
        priorityExponent=.5,
        epsilonInitial = 2,
        epsilonDecay = .9985,
        minFramesForTraining=2048,
        noisyLayers=False
    )
    testResults = a.execute()
    if len(allResults) > 0:
        allResults = np.vstack((allResults, testResults))
    else:
        allResults = np.array([testResults], dtype=object,ndmin=2)
    np.savetxt("./test-results/lr-1-8-vmax200-10-atom.txt",allResults,delimiter=",")








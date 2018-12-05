import MySQLdb
import os
from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")

experimentName = "prioritization"
env = gym.make('LunarLander-v2')
sess = tf.Session()
prioritization = np.random.random()
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
    episodesPerTest=1,
    numTestPeriods=40000,
    numTestsPerTestPeriod=30,
    maxRunningMinutes=10,
    episodeStepLimit=1024,
    intermediateTests=False,

    render=False,
    showGraph=False,

    # hyperparameters
    valueMin=-400.0,
    valueMax=100.0,
    numAtoms=10,
    maxMemoryLength=100000,
    batchSize=256,
    networkSize=[128, 128, 256],
    learningRate=2e-4,
    priorityExponent=prioritization,
    epsilonInitial = 2,
    epsilonDecay = .9986,
    minExploration = .05,
    maxExploration = .5,
    minFramesForTraining = 2048,
    maxGradientNorm = 5,
    noisyLayers = False
)
performance = a.execute()[0]
cur = db.cursor()
cur.execute("insert into experiments (label, x1, x2, x3, x4, y) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}')".format(experimentName, prioritization, 0, 0, 0, performance))
db.commit()
cur.close()
db.close()





















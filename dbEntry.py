import MySQLdb
import os
from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")

experimentName = "iqn-learning-rate-2"
env = gym.make('LunarLander-v2')
sess = tf.Session()
batchSize=256
learningRateLog = np.random.uniform(low=-3, high=-1)
learningRate = 1 * pow(10, learningRateLog)
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
    maxRunningMinutes=15,
    episodeStepLimit=1024,
    intermediateTests=False,

    render=False,
    showGraph=False,

    # hyperparameters
    maxMemoryLength=int(1e6),
    batchSize=256,
    learningRate=learningRate,
    priorityExponent= 0,
    epsilonInitial = 2,
    epsilonDecay = .997,
    minExploration = .01,
    maxExploration = .85,
    minFramesForTraining = 2048,
    maxGradientNorm = 5,
    preNetworkSize = [128, 128],
    postNetworkSize = [256],
    numQuantiles = 12,
    embeddingDimension = 24,
    kappa = 1.0,
    trainingIterations = 3
)
performance = a.execute()[0]
cur = db.cursor()
cur.execute("insert into experiments (label, x1, x2, x3, x4, y) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}')".format(experimentName, learningRate, 0, 0, 0, performance))
db.commit()
cur.close()
db.close()





















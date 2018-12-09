import MySQLdb
import os
from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")

experimentName = "standard-dqn-learning-rate"
env = gym.make('LunarLander-v2')
sess = tf.Session()
batchSize=256
learningRate = np.random.uniform(low=1e-5, high=1e-3)
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
    episodesPerTest=40000,
    numTestPeriods=1,
    numTestsPerTestPeriod=30,
    maxRunningMinutes=1,
    episodeStepLimit=1024,
    intermediateTests=False,

    render=False,
    showGraph=True,

    # hyperparameters
    maxMemoryLength=int(10e6),
    batchSize=batchSize,
    networkSize=[128, 128, 512],
    learningRate=learningRate,
    priorityExponent= 0,
    epsilonInitial = 2,
    epsilonDecay = .9995,
    minExploration = .05,
    maxExploration = .85,
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





















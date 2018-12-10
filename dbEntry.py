import MySQLdb
import os
from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")

experimentName = "dueling-dqn-prioritization"
env = gym.make('LunarLander-v2')
sess = tf.Session()
batchSize=256
prioritization = np.random.uniform(low=0, high=1)
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
    maxRunningMinutes=25,
    episodeStepLimit=1024,
    intermediateTests=False,

    render=False,
    showGraph=False,

    # hyperparameters
    maxMemoryLength=int(1e6),
    batchSize=256,
    networkSize=[128, 128],
    advantageNetworkSize=[512],
    valueNetworkSize=[512],
    learningRate=0.00625851,
    priorityExponent= prioritization,
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





















import MySQLdb
import os
from agent.agent import Agent
import numpy as np
import tensorflow as tf
import gym
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")

experimentName = "iqn-tau"
env = gym.make('LunarLander-v2')
sess = tf.Session()
tauExp = np.random.random(low=-1, high=-4)
tau = 1 * pow(10, tauExp)
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
    maxRunningMinutes=40,
    episodeStepLimit=1024,
    intermediateTests=False,

    render=False,
    showGraph=False,
    saveModel=False,
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
    tau = tau
)
performance = a.execute()[0]
cur = db.cursor()
cur.execute("insert into experiments (label, x1, x2, x3, x4, y) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}')".format(experimentName, tau, 0, 0, 0, performance))
db.commit()
cur.close()
db.close()





















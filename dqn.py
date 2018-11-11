from agent.agent import Agent
import tensorflow as tf
import gym

env = gym.make('LunarLander-v2')

sess = tf.Session()
a = Agent(
    sess=sess,
    env=env,
    episodeLimit=8192,
    numAvailableActions=4,
    numObservations=8,
    episodeStepLimit=1024,
    episodeSampleSize=20,
    valueMin=-400.0,
    valueMax=100.0,
    numAtoms=10,
    render=True,
    showGraph=True,

    maxMemoryLength=100000,
    gamma=1,
    batchSize=256,
    networkSize=[64,64,512],
    learningRate=3e-4,
    nStepUpdate=1,
    priorityExponent=.5,
    includeIntermediatePairs=False,
    updateTargetNetworkPeriod=1000,
    minFramesForTraining=2048,
    minFramesForTargetUpdate=2048,
    trainNetworkPeriod=4,
    testPeriod=100,
    epsilonInitial = 2,
    epsilonDecay = .998
)
a.execute()


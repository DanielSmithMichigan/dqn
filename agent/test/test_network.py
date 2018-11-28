from agent.network import Network
import random
import tensorflow as tf
import unittest
from agent import constants
from agent import util
import numpy as np
import sys

# def trainNetworkTo(sess, network, bestAction, numTrainingIterations, maxAvailableActions):
#     for i in range(numTrainingIterations):
#         action = random.randint(0, maxAvailableActions - 1)
#         reward = 1 if action == bestAction else 0
#         memory = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
#         memory[constants.STATE] = [random.random()]
#         memory[constants.ACTION] = action
#         memory[constants.REWARD] = reward
#         memory[constants.NEXT_STATE] = [random.random()]
#         memory[constants.GAMMA] = 0
#         network.trainAgainst([memory, memory])

# def testNetworkForAction(assertEqual, sess, network, bestAction, numTestingIterations):
#     for i in range(numTestingIterations):
#         chosenAction, qValues = sess.run([network.chosenAction, network.qValues], feed_dict={
#             network.actionInput: [0,1,0],
#             network.environmentInput: [
#                 [random.random()],
#                 [random.random()],
#                 [random.random()]
#             ]
#         })
#         for i in range(len(chosenAction)):
#             assertEqual(chosenAction[i], bestAction)

# class TestNetwork(unittest.TestCase):
    # def testSupport(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-1",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=1,
    #             learningRate=1e-5,
    #             numAtoms=3,
    #             valueMin=0.0,
    #             valueMax=6.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         support = sess.run(n.support)
    #         np.testing.assert_equal(support, [0,3,6])
    # def testQValues(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-2",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=1,
    #             learningRate=1e-5,
    #             numAtoms=51,
    #             valueMin=-200.0,
    #             valueMax=200.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         probabilities = sess.run(n.probabilities, feed_dict={
    #             n.environmentInput: [[0.0]]
    #         })
    #         self.assertEqual(np.shape(probabilities), (1, 1, 51))
    # def testActionProbabilities(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-3",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=3,
    #             learningRate=1e-5,
    #             numAtoms=4,
    #             valueMin=-200.0,
    #             valueMax=200.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         actionLogits, logits = sess.run([n.actionLogits, n.logits], feed_dict={
    #             n.environmentInput: [[0.0], [0.2], [-0.1]],
    #             n.actionInput: [2, 0, 1]
    #         })
    #         self.assertEqual(np.shape(actionLogits), (3, 4))
    #         np.testing.assert_equal(actionLogits[0], logits[0][2])
    #         np.testing.assert_equal(actionLogits[1], logits[1][0])
    #         np.testing.assert_equal(actionLogits[2], logits[2][1])
    # def testQValues(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-4",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=3,
    #             learningRate=1e-5,
    #             numAtoms=3,
    #             valueMin=0.0,
    #             valueMax=6.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         qValues, probabilities, support = sess.run([n.qValues, n.probabilities, n.support], feed_dict={
    #             n.environmentInput: [[0.0], [1.0], [0.5]]
    #         })
    #         self.assertEqual(np.shape(qValues), (3, 3))
    #         for _batchIndex in range(3):
    #             for _actionIndex in range(3):
    #                 probabilityDistribution = probabilities[_batchIndex][_actionIndex]
    #                 self.assertEqual(np.shape(probabilityDistribution), (3,))
    #                 distributionTotal = 0.0
    #                 for _supportIndex in range(3):
    #                     distributionTotal = distributionTotal + probabilityDistribution[_supportIndex] * support[_supportIndex]
    #                 self.assertAlmostEqual(distributionTotal, qValues[_batchIndex][_actionIndex] + 0.0, 5)
    # def testMaxQ(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-5",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=3,
    #             learningRate=1e-5,
    #             numAtoms=3,
    #             valueMin=0.0,
    #             valueMax=6.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         qValues, chosenAction, maxQ = sess.run([n.qValues, n.chosenAction, n.maxQ], feed_dict={
    #             n.environmentInput: [[0.0], [1.0], [0.5]]
    #         })
    #         for _batchIndex in range(3):
    #             batchQs = qValues[_batchIndex]
    #             maxBatchQ = np.max(batchQs)
    #             maxBatchQIndex = np.argmax(batchQs)
    #             self.assertEqual(maxBatchQ, maxQ[_batchIndex])
    #             self.assertEqual(maxBatchQIndex, chosenAction[_batchIndex])
    # def testChosenActionProbabilities(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-6",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16],
    #             numAvailableActions=3,
    #             learningRate=1e-5,
    #             numAtoms=3,
    #             valueMin=0.0,
    #             valueMax=6.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         probabilities, chosenAction, chosenActionProbabilities = sess.run([n.probabilities, n.chosenAction, n.chosenActionProbabilities], feed_dict={
    #             n.environmentInput: [[0.0], [1.0], [0.5]]
    #         })
    #         for _batchIndex in range(3):
    #             np.testing.assert_equal(probabilities[_batchIndex][chosenAction[_batchIndex]], chosenActionProbabilities[_batchIndex])
    # def testTrainForDistribution(self):
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-7",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[64,64,64],
    #             numAvailableActions=1,
    #             learningRate=1e-3,
    #             numAtoms=51,
    #             valueMin=-10.0,
    #             valueMax=10.0
    #         )
    #         sess.run(tf.global_variables_initializer())
    #         dist = np.zeros(51)
    #         dist[0] = 1
    #         for i in range(5000):
    #             _, loss = sess.run([n.trainingOperation, n.loss], feed_dict={
    #                 n.actionInput: [0],
    #                 n.targetProbabilities: [dist],
    #                 n.environmentInput: [
    #                     [np.random.random()]
    #                 ]
    #             })
    #         loss = sess.run([n.loss], feed_dict={
    #             n.actionInput: [0],
    #             n.targetProbabilities: [dist],
    #             n.environmentInput: [
    #                 [np.random.random()]
    #             ]
    #         })
    #         self.assertEqual(loss[0] < 1e-3, True)
    # def testTrainTwoNetworks(self):
    #     with tf.Session() as sess:
    #         learnedNetwork = Network(
    #             name="network-8",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[64,64,64],
    #             numAvailableActions=1,
    #             learningRate=1e-3,
    #             numAtoms=51,
    #             valueMin=-10.0,
    #             valueMax=10.0
    #         )
    #         targetNetwork = Network(
    #             name="network-9",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[64,64,64],
    #             numAvailableActions=1,
    #             learningRate=1e-3,
    #             numAtoms=51,
    #             valueMin=-10.0,
    #             valueMax=10.0
    #         )
    #         targetNetwork.makeDuplicationOperation(learnedNetwork.networkParams)
    #         learnedNetwork.makeDuplicationOperation(targetNetwork.networkParams)
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(learnedNetwork.duplicateOtherNetwork)
    #         batchSize = 32
    #         trainingPerTargetUpdate = 256
    #         targetUpdates = 12
    #         for i in range(targetUpdates):
    #             for j in range(trainingPerTargetUpdate):
    #                 batch = []
    #                 for k in range(batchSize):
    #                     memory = np.zeros(constants.NUM_MEMORY_ENTRIES, dtype=object)
    #                     memory[constants.ACTION] = 0
    #                     memory[constants.STATE] = [np.random.random()]
    #                     memory[constants.REWARD] = 1
    #                     memory[constants.NEXT_STATE] = [np.random.random()]
    #                     memory[constants.GAMMA] = 0
    #                     memory[constants.IS_TERMINAL] = True
    #                     batch.append(memory)
    #                 targetProbabilities = targetNetwork.getTargetDistributions(batch)
    #                 _, loss = sess.run([learnedNetwork.trainingOperation, learnedNetwork.loss], feed_dict={
    #                     learnedNetwork.actionInput: util.getColumn(batch, constants.ACTION),
    #                     learnedNetwork.targetProbabilities: targetProbabilities,
    #                     learnedNetwork.environmentInput: util.getColumn(batch, constants.STATE)
    #                 })
    #                 print("LOSS: ",np.mean(loss))
    #             sess.run(targetNetwork.duplicateOtherNetwork)
    #             print("TARGET UPDATED")
            # self.assertEqual(loss[0] < 1e-3, True)
    # def testTrainIt(self):
    #     NUM_TRAINING_ITERATIONS = 100
    #     with tf.Session() as sess:
    #         n = Network(
    #             name="network-7",
    #             sess=sess,
    #             numObservations=1,
    #             networkSize=[16,16],
    #             numAvailableActions=5,
    #             learningRate=1e-3,
    #             numAtoms=4,
    #             valueMin=0.0,
    #             valueMax=10
    #         )
    #         for testEpisode in range(2):
    #             for bestAction in range(5):
    #                 sess.run(tf.global_variables_initializer())
    #                 trainNetworkTo(sess, n, bestAction, NUM_TRAINING_ITERATIONS, 5)
    #                 testNetworkForAction(self.assertEqual, sess, n, bestAction, NUM_TRAINING_ITERATIONS)
#     def testAssignment(self):
#         NUM_TRAINING_ITERATIONS = 100
#         with tf.Session() as sess:
#             n = Network(
#                 name="network-a",
#                 sess=sess,
#                 numObservations=1,
#                 networkSize=[16,16],
#                 numAvailableActions=5,
#                 learningRate=1e-3,
#                 numAtoms=4,
#                 valueMin=0.0,
#                 valueMax=10
#             )
#             n2 = Network(
#                 name="network-b",
#                 sess=sess,
#                 numObservations=1,
#                 networkSize=[16,16],
#                 numAvailableActions=5,
#                 learningRate=1e-3,
#                 numAtoms=4,
#                 valueMin=0.0,
#                 valueMax=10
#             )
#             n.makeDuplicationOperation(n2.networkParams)
#             sess.run(tf.global_variables_initializer())
#             trainNetworkTo(sess, n, 1, NUM_TRAINING_ITERATIONS, 5)
#             testNetworkForAction(self.assertEqual, sess, n, 1, NUM_TRAINING_ITERATIONS)
#             trainNetworkTo(sess, n2, 2, NUM_TRAINING_ITERATIONS, 5)
#             testNetworkForAction(self.assertEqual, sess, n2, 2, NUM_TRAINING_ITERATIONS)
#             sess.run(n.duplicateOtherNetwork)
#             testNetworkForAction(self.assertEqual, sess, n, 2, NUM_TRAINING_ITERATIONS)
#             testNetworkForAction(self.assertEqual, sess, n2, 2, NUM_TRAINING_ITERATIONS)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

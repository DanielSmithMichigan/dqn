import numpy as np
import tensorflow as tf
from c51.distribution import buildTargets
from .noise import noisy_dense_layer
from . import constants
from . import util
import math
slim = tf.contrib.slim

class Network:
    def __init__(self,
            name,
            sess,
            numObservations,
            preNetworkSize,
            postNetworkSize,
            numQuantiles,
            numAvailableActions,
            embeddingDimension,
            learningRate,
            maxGradientNorm,
            batchSize,
            kappa,
            targetNetwork=None
        ):
        self.name = name
        self.sess = sess
        self.numAvailableActions = numAvailableActions
        self.numObservations = numObservations
        self.preNetworkSize = preNetworkSize
        self.postNetworkSize = postNetworkSize
        self.numQuantiles = numQuantiles
        self.embeddingDimension = embeddingDimension
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.kappa = kappa
        self.maxGradientNorm = maxGradientNorm
        self.targetNetwork = targetNetwork
        self.losses = []
        self.build()
    def build(self):
        with tf.variable_scope(self.name):
            self.weightsInitializer = slim.variance_scaling_initializer(factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)
            self.environmentInput = prevLayer = tf.placeholder(tf.float32, [None, self.numObservations], "EnvironmentInput")
            # Shape: batchSize x numObservations
            for i in range(len(self.preNetworkSize)):
                prevLayer = tf.layers.dense(inputs=prevLayer, units=self.preNetworkSize[i], activation=tf.nn.leaky_relu, kernel_initializer=self.weightsInitializer, name="pre_hidden_"+str(i))
            # Shape: batchSize x N
            self.buildEmbedding()
            # Shape: batchSize x numQuantiles x embeddingDimension
            self.embeddedQuantiles = tf.layers.dense(inputs=self.comparableQuantiles, units=self.preNetworkSize[-1], activation=tf.nn.leaky_relu, kernel_initializer=self.weightsInitializer)
            # Shape: batchSize x numQuantiles x N
            prevLayer = tf.reshape(prevLayer, [-1, 1, self.preNetworkSize[-1]])
            # Shape: batchSize x 1            x N
            self.firstJoinedLayer = prevLayer = self.embeddedQuantiles * prevLayer
            # Shape: batchSize x numQuantiles x N
            for i in range(len(self.postNetworkSize)):
                prevLayer = tf.layers.dense(inputs=prevLayer, units=self.postNetworkSize[i], activation=tf.nn.leaky_relu, kernel_initializer=self.weightsInitializer, name="post_hidden_"+str(i))
            # Shape: batchSize x numQuantiles x N
            self.quantileValuesActions = tf.layers.dense(inputs=prevLayer, units=self.numAvailableActions, kernel_initializer=self.weightsInitializer)
            # Shape: batchSize x numQuantiles x numActions
            self.quantileValuesAverageAction = tf.reshape(tf.reduce_mean(self.quantileValuesActions, axis=2), [-1, self.numQuantiles, 1])
            # Shape: batchSize x numQuantiles x 1
            self.quantileValuesAdvantages = self.quantileValuesActions - self.quantileValuesAverageAction
            # Shape: batchSize x numQuantiles x numActions
            self.quantileValuesValue = tf.layers.dense(inputs=prevLayer, units=1, kernel_initializer=self.weightsInitializer)
            # Shape: batchSize x numQuantiles x 1
            self.quantileValues = self.quantileValuesAdvantages + self.quantileValuesValue
            # Shape: batchSize x numQuantiles x numActions
            self.quantileValues = tf.transpose(self.quantileValues, [0, 2, 1])
            # Shape: batchSize x numActions x numQuantiles
        self.networkParams = tf.trainable_variables(scope=self.name)
        self.buildMaxQHead()
        self.buildIndexedQuantiles()
        if self.targetNetwork:
            self.buildTrainingOperation()
    def buildEmbedding(self):
        self.quantileThresholds = tf.placeholder(tf.float32, [None, self.numQuantiles], "InputRandoms")
        # Shape: batchSize x numQuantiles
        self.reshapedQuantileThresholdsBox = tf.reshape(self.quantileThresholds, [-1, self.numQuantiles, 1])
        # Shape: batchSize x numQuantiles x 1
        self.quantilesEmbeddingBox = tf.tile(self.reshapedQuantileThresholdsBox, [1, 1, self.embeddingDimension])
        # Shape: batchSize x numQuantiles x embeddingDimension
        self.cosineTimestep = tf.reshape(tf.cast(tf.range(self.embeddingDimension) + 1, tf.float32), [1, 1, self.embeddingDimension])
        # Shape: 1         x 1            x embeddingDimension
        self.tiledQuantilesPreCos = self.quantilesEmbeddingBox * self.cosineTimestep
        # Shape: batchSize x numQuantiles x embeddingDimension
        self.tiledQuantiles = tf.math.cos(self.tiledQuantilesPreCos * math.pi)
        # Shape: batchSize x numQuantiles x embeddingDimension
        self.comparableQuantiles = self.tiledQuantiles
        # Shape: batchSize x numQuantiles x embeddingDimension
    def buildMaxQHead(self):
        self.reversedThresholds = tf.constant(1.0) - self.quantileThresholds
        # Shape: batchSize x numQuantiles
        self.comparableReversedThresholds = tf.reshape(self.reversedThresholds, [-1, 1, self.numQuantiles])
        # Shape: batchSize x 1 x numQuantiles
        self.individualQValues = self.comparableReversedThresholds * self.quantileValues
        # Shape: batchSize x numActions x numQuantiles
        self.qValues = tf.reduce_mean(self.individualQValues, axis=2)
        # Shape: batchSize x numActions
        self.maxQ = tf.reduce_max(self.qValues, axis=1)
        self.chosenAction = tf.argmax(self.qValues, axis=1)
        batchIndices = tf.cast(tf.range(tf.shape(self.qValues)[0]), tf.int64)
        self.chosenActionQValue = tf.gather_nd(self.qValues, tf.stack([batchIndices, tf.cast(self.chosenAction, tf.int64)], axis=1))
    def buildIndexedQuantiles(self):
        self.actionInput = tf.placeholder(tf.int32, [None,], name="ActionInput")
        # Shape: batchSize
        self.indexedQuantiles = tf.gather_nd(self.quantileValues, tf.cast(tf.stack([tf.range(tf.shape(self.quantileValues)[0]), self.actionInput], axis=1), tf.int64))
        # Shape: batchSize x numQuantiles
    def buildTrainingOperation(self):
        self.gammas = tf.placeholder(tf.float32, [None,], name="Gammas")
        # Shape: batchSize
        self.observedRewards = tf.placeholder(tf.float32, [None,], name="ObservedRewards")
        # Shape: batchSize
        self.memoryPriority = tf.placeholder(tf.float32, [None,], name="MemoryPriority")
        # Shape: batchSize
        comparableRewards = tf.reshape(self.observedRewards, [-1, 1])
        # Shape: batchSize x 1
        comparableGammas = tf.reshape(self.gammas, [-1, 1])
        # Shape: batchSize x 1
        self.quantileDistance = (self.targetNetwork.indexedQuantiles * comparableGammas + comparableRewards) - self.indexedQuantiles
        # Shape: batchSize x numQuantiles
        self.absQuantileDistance = tf.abs(self.quantileDistance)
        # Shape: batchSize x numQuantiles
        self.minorQuantileError = tf.to_float(self.absQuantileDistance <= self.kappa) * 0.5 * self.absQuantileDistance ** 2
        # Shape: batchSize x numQuantiles
        self.majorQuantileError = tf.to_float(self.absQuantileDistance > self.kappa) * self.kappa * (self.absQuantileDistance - 0.5 * self.kappa)
        # Shape: batchSize x numQuantiles
        self.totalQuantileError = self.minorQuantileError + self.majorQuantileError
        # Shape: batchSize x numQuantiles
        self.belowQuantile = tf.to_float(comparableRewards < self.indexedQuantiles)
        # Shape: batchSize x numQuantiles
        self.sizedQuantiles = tf.abs(self.quantileThresholds - self.belowQuantile)
        # Shape: batchSize x numQuantiles
        self.quantileRegressionLoss = self.sizedQuantiles * self.totalQuantileError / self.kappa
        # Shape: batchSize x numQuantiles
        self.batchwiseLoss = tf.reduce_sum(self.quantileRegressionLoss, axis=1)
        # Shape: batchSize
        self.proportionedLoss = self.batchwiseLoss / self.memoryPriority
        # Shape: batchSize
        self.finalLoss = tf.reduce_mean(self.proportionedLoss)
        # Shape: 1
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        gradients, variables = zip(*self.optimizer.compute_gradients(self.finalLoss))
        self.gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradientNorm)
        self.trainingOperation = self.optimizer.apply_gradients(zip(self.gradients, variables))
    def buildSoftCopyOperation(self, networkParams, tau):
        return [tf.assign(t, (1 - tau) * t + tau * e) for t, e in zip(self.networkParams, networkParams)]
    def trainAgainst(self, memoryUnits):
        actions = util.getColumn(memoryUnits, constants.ACTION)
        quantileThresholds = np.random.uniform(low=0.0, high=1.0, size=(self.batchSize, self.numQuantiles))
        nextActions = self.sess.run(self.chosenAction, feed_dict={
            self.environmentInput: util.getColumn(memoryUnits, constants.NEXT_STATE),
            self.quantileThresholds: quantileThresholds
        })
        targets, predictions, batchwiseLoss, finalLoss, _ = self.sess.run([self.observedRewards, self.chosenActionQValue, self.batchwiseLoss, self.finalLoss, self.trainingOperation], feed_dict={
            self.environmentInput: util.getColumn(memoryUnits, constants.STATE),
            self.memoryPriority: util.getColumn(memoryUnits, constants.PRIORITY),
            self.actionInput: actions,
            self.observedRewards: util.getColumn(memoryUnits, constants.REWARD),
            self.gammas: util.getColumn(memoryUnits, constants.GAMMA),
            self.quantileThresholds: quantileThresholds,
            self.targetNetwork.environmentInput: util.getColumn(memoryUnits, constants.NEXT_STATE),
            self.targetNetwork.actionInput: nextActions,
            self.targetNetwork.quantileThresholds: quantileThresholds
        })
        self.losses.append(finalLoss)
        for i in range(len(memoryUnits)):
            memoryUnits[i][constants.LOSS] = batchwiseLoss[i]
        return targets, predictions, actions
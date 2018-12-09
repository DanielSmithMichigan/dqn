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
            networkSize,
            advantageNetworkSize,
            valueNetworkSize,
            numAvailableActions,
            learningRate,
            noisyLayers,
            maxGradientNorm,
            batchSize
        ):
        self.name = name
        self.sess = sess
        self.noisyLayers = noisyLayers
        self.numAvailableActions = numAvailableActions
        self.numObservations = numObservations
        self.networkSize = networkSize
        self.valueNetworkSize = valueNetworkSize
        self.advantageNetworkSize = advantageNetworkSize
        self.learningRate = learningRate
        self.maxGradientNorm = maxGradientNorm
        self.batchSize = batchSize
        self.losses = []
        self.build()
    def build(self):
        weights_initializer = slim.variance_scaling_initializer(factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)
        with tf.variable_scope(self.name):
            self.environmentInput = tf.placeholder(tf.float32, [None, self.numObservations], "EnvironmentInput")
            prevLayer = self.environmentInput
            for i in range(len(self.networkSize)):
                prevLayer = tf.layers.dense(inputs=prevLayer, units=self.networkSize[i], activation=tf.nn.leaky_relu, kernel_initializer=weights_initializer, name="hidden_"+str(i))
            prevLayerAdvantage = prevLayer
            prevLayerValue = prevLayer
            for i in range(len(self.advantageNetworkSize)):
                prevLayerAdvantage = tf.layers.dense(inputs=prevLayerAdvantage, units=self.advantageNetworkSize[i], activation=tf.nn.leaky_relu, kernel_initializer=weights_initializer, name="advantage_"+str(i))
            self.advantage = tf.layers.dense(inputs=prevLayer, units=self.numAvailableActions, kernel_initializer=weights_initializer, name="advantage-logits")
            self.advantage = tf.reshape(self.advantage, [-1, self.numAvailableActions])
            for i in range(len(self.valueNetworkSize)):
                prevLayerValue = tf.layers.dense(inputs=prevLayerValue, units=self.valueNetworkSize[i], activation=tf.nn.leaky_relu, kernel_initializer=weights_initializer, name="value_"+str(i))
            self.value = tf.layers.dense(inputs=prevLayer, units=1, kernel_initializer=weights_initializer, name="value-logits")
            self.averageAdvantage = tf.reduce_mean(self.advantage)
            self.qValues = self.value + (self.advantage - self.averageAdvantage)
            self.buildActionProbabilityHead()
            self.buildMaxQHead()
            self.buildTrainingOperation()
        self.networkParams = tf.trainable_variables(scope=self.name)
    def buildMaxQHead(self):
        self.maxQ = tf.reduce_max(self.qValues, axis=1)
        self.chosenAction = tf.argmax(self.qValues, axis=1)
        self.chosenActionQValue = self.indexInto(self.qValues, self.chosenAction)
    def buildActionProbabilityHead(self):
        self.actionInput = tf.placeholder(tf.int32, [None,], name="ActionInput")
        self.actionValues = self.indexInto(self.qValues, self.actionInput)
    def buildTrainingOperation(self):
        self.targetQValues = tf.placeholder(tf.float32, [None,])
        self.lossPriority = tf.placeholder(tf.float32, [None,])
        self.lossWeights = (1.0 / self.lossPriority)
        self.pairwiseLoss = tf.squared_difference(self.targetQValues, self.actionValues) * self.lossWeights
        self.meanLoss = tf.reduce_mean(self.pairwiseLoss)
        self.alternateLoss = tf.losses.mean_squared_error(
            labels=self.targetQValues,
            predictions=self.actionValues,
            weights=self.lossWeights
        )
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        gradients, variables = zip(*self.optimizer.compute_gradients(self.meanLoss))
        self.gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradientNorm)
        self.trainingOperation = self.optimizer.apply_gradients(zip(self.gradients, variables))
    def indexInto(self, ary, indexes):
        batchIndices = tf.range(tf.shape(indexes)[0])
        actionProbabilityIndices = tf.stack([tf.cast(batchIndices, tf.int64), tf.cast(indexes, tf.int64)], axis=1)
        return tf.gather_nd(ary, actionProbabilityIndices)
    def trainAgainst(self, memoryUnits):
        actions = util.getColumn(memoryUnits, constants.ACTION)
        targets, predictions, pairwiseLoss, meanLoss, alternateLoss, _ = self.sess.run([self.targetQValues, self.chosenActionQValue, self.pairwiseLoss, self.meanLoss, self.alternateLoss, self.trainingOperation], feed_dict={
            self.environmentInput: util.getColumn(memoryUnits, constants.STATE),
            self.lossPriority: util.getColumn(memoryUnits, constants.PRIORITY),
            self.actionInput: actions,
            self.targetQValues: util.getColumn(memoryUnits, constants.REWARD)
        })
        self.losses.append(np.mean(meanLoss))
        for i in range(len(memoryUnits)):
            memoryUnits[i][constants.LOSS] = pairwiseLoss[i]
        return targets, predictions, actions
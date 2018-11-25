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
            numAvailableActions,
            learningRate,
            numAtoms,
            valueMin,
            valueMax,
            noisyLayers,
            maxGradientNorm
        ):
        self.name = name
        self.sess = sess
        self.noisyLayers = noisyLayers
        self.numAvailableActions = numAvailableActions
        self.numObservations = numObservations
        self.networkSize = networkSize
        self.learningRate = learningRate
        self.numAtoms = numAtoms
        self.maxGradientNorm = maxGradientNorm
        self.support = tf.linspace(valueMin, valueMax, numAtoms)
        self.losses = []
        self.build()
    def build(self):
        weights_initializer = slim.variance_scaling_initializer(factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)
        with tf.variable_scope(self.name):
            self.environmentInput = tf.placeholder(tf.float32, [None, self.numObservations], "EnvironmentInput")
            prevLayer = self.environmentInput
            if (self.noisyLayers):
                for i in range(len(self.networkSize)):
                    prevLayer = noisy_dense_layer(inputs=prevLayer, num_units=self.networkSize[i])
                    prevLayer = tf.nn.leaky_relu(prevLayer)
                self.logits = noisy_dense_layer(inputs=prevLayer, num_units=self.numAvailableActions * self.numAtoms)
            else:
                for i in range(len(self.networkSize)):
                    prevLayer = tf.layers.dense(inputs=prevLayer, units=self.networkSize[i], activation=tf.nn.leaky_relu, kernel_initializer=weights_initializer, name="hidden_"+str(i))
                self.logits = tf.layers.dense(inputs=prevLayer, units=self.numAvailableActions * self.numAtoms, kernel_initializer=weights_initializer, name="logits")
            self.logits = tf.reshape(self.logits, [-1, self.numAvailableActions, self.numAtoms])
            self.probabilities = tf.nn.softmax(self.logits, axis=2)
            self.buildActionProbabilityHead()
            self.buildMaxQHead()
            self.buildProjectOperation()
            self.buildTrainingOperation()
        self.networkParams = tf.trainable_variables(scope=self.name)
    def buildMaxQHead(self):
        self.qValues = tf.reduce_sum(self.support * self.probabilities, axis=2)
        self.maxQ = tf.reduce_max(self.qValues, axis=1)
        self.chosenAction = tf.argmax(self.qValues, axis=1)
        self.chosenActionProbabilities = self.indexInto(self.probabilities, self.chosenAction)
    def buildActionProbabilityHead(self):
        self.actionInput = tf.placeholder(tf.int32, [None,], name="ActionInput")
        self.actionLogits = self.indexInto(self.logits, self.actionInput)
    def buildTrainingOperation(self):
        self.targetDistributions = tf.placeholder(tf.float32, [None, self.numAtoms])
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.targetDistributions,logits=tf.clip_by_value(self.actionLogits, -1, 2))
        self.mean_loss = tf.reduce_mean(self.loss)
        # self.trainingOperation = tf.train.AdamOptimizer(self.learningRate).minimize(self.mean_loss)
        self.optimizer = tf.train.AdamOptimizer(self.learningRate)
        gradients, variables = zip(*self.optimizer.compute_gradients(self.mean_loss))
        self.gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradientNorm)
        self.trainingOperation = self.optimizer.apply_gradients(zip(self.gradients, variables))
    def indexInto(self, ary, indexes):
        batchIndices = tf.range(tf.shape(indexes)[0])
        actionProbabilityIndices = tf.stack([tf.cast(batchIndices, tf.int64), tf.cast(indexes, tf.int64)], axis=1)
        return tf.gather_nd(ary, actionProbabilityIndices)
    def makeDuplicationOperation(self, networkParams):
        self.duplicateOtherNetwork = [tf.assign(t, e) for t, e in zip(self.networkParams, networkParams)]
    def buildProjectOperation(self):
        self.rewardsInput = tf.placeholder(tf.float32, [None, ])
        self.gammasInput = tf.placeholder(tf.float32, [None, ])
        self.targetDistributionOp = buildTargets(self.support, self.chosenActionProbabilities, self.rewardsInput, self.gammasInput)
    def getTargetDistributions(self, memoryUnits):
        return self.sess.run(self.targetDistributionOp, feed_dict={
            self.environmentInput: util.getColumn(memoryUnits, constants.NEXT_STATE),
            self.rewardsInput: util.getColumn(memoryUnits, constants.REWARD),
            self.gammasInput: util.getColumn(memoryUnits, constants.GAMMA)
        })
    def trainAgainst(self, memoryUnits, support):
        actions = util.getColumn(memoryUnits, constants.ACTION)
        targetDistributions = []
        for i in memoryUnits:
            targetDistribution = np.zeros(self.numAtoms)
            lowestDistIndex = -1
            lowestDist = 1000000000
            for j in range(len(support)):
                dist = abs(i[constants.REWARD] - support[j])
                if (dist < lowestDist):
                    lowestDist = dist
                    lowestDistIndex = j
            targetDistribution[lowestDistIndex] = 1
            targetDistributions.append(targetDistribution)

        targets, predictions, loss, _ = self.sess.run([self.targetDistributions, self.actionLogits, self.loss, self.trainingOperation], feed_dict={
            self.environmentInput: util.getColumn(memoryUnits, constants.STATE),
            self.actionInput: actions,
            self.targetDistributions: targetDistributions
        })
        self.losses.append(np.mean(loss))
        for i in range(len(memoryUnits)):
            memoryUnits[i][constants.LOSS] = loss[i]
        return targets, predictions, actions
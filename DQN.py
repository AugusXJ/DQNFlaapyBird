# -*- coding: utf-8 -*-
"""
@Time    : 05/07/18 16:18
@Author  : XJH
"""

import FlappyBird
import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import os
import time


class Qnetwork():
    def __init__(self, h_size):
        self.scalarInput = tf.placeholder(shape=[None, 147456], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 288, 512, 3])
        self.conv1 = tf.contrib.layers.convolution2d(
            inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
            padding='VALID', biases_initializer=None
        )
        self.conv2 = tf.contrib.layers.convolution2d(
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
            padding='VALID', biases_initializer=None
        )
        self.conv3 = tf.contrib.layers.convolution2d(
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
            padding='VALID', biases_initializer=None
        )
        self.conv4 = tf.contrib.layers.convolution2d(
            inputs=self.conv3, num_outputs=512, kernel_size=[7, 7], stride=[1, 1],
            padding='VALID', biases_initializer=None
        )
        # Duel DQN
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)       # 拆分成2段，维度是第三维
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        self.AW = tf.Variable(tf.random_normal([h_size//2, env.actions]))       # streamA全连接层权重
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices=1,
                                                                            keep_dims=True))    # 合并Q值
        self.predict = tf.argmax(self.Qout, 1)
        # Double DQN
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        # loss
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=50000):
        """
        经验回放
        :param buffer_size:
        """
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx + total_vars//2].assign((var.value() * tau) +
                                                            (1-tau) * tfVars[idx + total_vars//2].value()))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

def processState(states):
    """
    扁平化为1维向量
    :param states:
    :return:
    """
    return np.reshape(states, [147456])

batch_size = 32
update_freq = 4             # 每隔多少步更新一次模型参数
y = .99                     # Q值得衰减系数
startE = 1.                  # 初始时执行随机行为的概率
endE = 0.1                  # 最终执行随机行为的概率
anneling_steps = 10000.     # 从初始随机行为到最终随机行为所需步数
num_episodes = 10000        # 总共需要多少次游戏
pre_train_steps = 10000     # 正式使用DQN选择action前需要多少步随机action
max_epLength = 50           # 每个episode进行多少步action
load_model = False          # 是否读取之前训练的模型
path = "./dqn"              # 模型存储的路径
h_size = 512                # DQN网络最后的全连接层隐含节点数
tau = 0.001                 # target DQN向主DQN学习的速率


if __name__ == '__main__':
    env = FlappyBird.env()
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)
    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, tau)
    myBuffer = experience_buffer()                          # 创建buffer对象
    e = startE
    stepDrop = (startE - endE) / anneling_steps

    rList = []
    total_steps = 0

    saver = tf.train.Saver()
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        if load_model is True:
            print('Loading Model ......')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(init)
        updateTarget(targetOps, sess)
        for i in range(num_episodes+1):
            episodeBuffer = experience_buffer()
            s = env.reset()
            s = processState(s)
            d = False                       # done标记
            rAll = 0                        # 总reward
            j = 0                           # 步数
            while j < max_epLength:
                j += 1
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = np.random.randint(0, 4)
                else:
                    a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                s1, r, d = env.step(a)
                s1 = processState(s1)
                total_steps += 1
                episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                    if total_steps % update_freq == 0:                                # 开始训练
                        trainBatch = myBuffer.sample(batch_size)
                        A = sess.run(mainQN.predict,
                                     feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})       # 主模型的action
                        Q = sess.run(targetQN.Qout,
                                     feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})     # 所有action的Q
                        doubleQ = Q[range(batch_size), A]
                        targetQ = trainBatch[:, 2] + y * doubleQ
                        _ = sess.run(mainQN.updateModel, feed_dict={
                            mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                            mainQN.targetQ: targetQ,
                            mainQN.actions: trainBatch[:, 1]
                        })
                        updateTarget(targetOps, sess)
                rAll += r
                s = s1
                if d is True:
                    break
            myBuffer.add(episodeBuffer.buffer)
            rList.append(rAll)
            if i > 0 and i % 25 == 0:
                print('episode', i, ', average reward of last 25 episode', np.mean(rList[-25:]))
            if i > 0 and i % 1000 == 0:
                saver.save(sess, path + '/model-' + str(i) + '.cpkt')
                print("Saved Model")
        saver.save(sess, path + 'model-' + str(i) + '.cpkt')
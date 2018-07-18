# -*- coding: utf-8 -*-
"""
@Time    : 05/07/18 16:18
@Author  : XJH
"""

import FlappyBird
import numpy as np
import random
import tensorflow as tf
import os
import time
from PIL import Image

actions = 2

class Qnetwork():
    def __init__(self, h_size):
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, img_height, img_weight, 3])
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
        self.AW = tf.Variable(tf.random_normal([h_size//2, actions]))       # streamA全连接层权重
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices=1,
                                                                            keep_dims=True))    # 合并Q值
        self.predict = tf.argmax(self.Qout, 1)
        # Double DQN
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, actions, dtype=tf.float32)
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
    img = Image.fromarray(np.uint8(states))
    img = img.resize((img_height, img_weight))
    states = np.asarray(img)
    return np.reshape(states, [21168])

batch_size = 32
update_freq = 4             # 每隔多少步更新一次模型参数
y = .99                     # Q值得衰减系数
startE = 1.                  # 初始时执行随机行为的概率
endE = 0.1                  # 最终执行随机行为的概率
anneling_episodes = 500.     # 从初始随机行为到最终随机行为所需eposide
num_episodes = 10000        # 总共需要多少次游戏
pre_train_steps = 10000     # 正式使用DQN选择action前需要多少步随机action
max_epLength = 50000           # 每个episode进行多少步action
load_model = True          # 是否读取之前训练的模型
path = "./dqn"              # 模型存储的路径
h_size = 512                # DQN网络最后的全连接层隐含节点数
tau = 0.001                 # target DQN向主DQN学习的速率
img_height = 84
img_weight = 84
stepDrop = (startE - endE) / anneling_episodes
e = endE.assign(endE - stepDrop)
drop_e = tf.subtract(e, tf.constant(stepDrop))
total_steps = tf.Variable(initial_value=0, trainable=False)
step_plus = total_steps.assign_add(1)

def extract_step(path):
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])


def loader(saver, session, load_dir):
    if tf.gfile.Exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            prev_step = extract_step(ckpt.model_checkpoint_path)

        else:
            tf.gfile.DeleteRecursively(load_dir)
            tf.gfile.MakeDirs(load_dir)
            prev_step = 0
    else:
        tf.gfile.MakeDirs(load_dir)
        prev_step = 0
    return prev_step

def test():
    env = FlappyBird.GameEnv()
    mainQN = Qnetwork(h_size)
    init = tf.global_variables_initializer()
    # trainables = tf.trainable_variables()
    # targetOps = updateTargetGraph(trainables, tau)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # updateTarget(targetOps, sess)
        print('Loading Model ......')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0  # 总reward
        j = 0  # 步数
        while j < max_epLength:
            j += 1
            a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            s1, r, d = env.step(a)
            env.render()
            s1 = processState(s1)
            rAll += r
            s = s1
            if d is True:
                break
        print(rAll)

def train():
    env = FlappyBird.GameEnv()
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)
    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, tau)
    myBuffer = experience_buffer()                          # 创建buffer对象

    rList = []



    saver = tf.train.Saver()
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        last_eposide = 0
        if load_model is True:
            print('Loading Model ......')
            last_eposide = loader(saver, sess, path)
        sess.run(init)
        updateTarget(targetOps, sess)
        for i in range(last_eposide, num_episodes+1):
            # print("eposide {}/{}".format(i, num_episodes))
            episodeBuffer = experience_buffer()
            s = env.reset()
            s = processState(s)
            d = False                       # done标记
            rAll = 0                        # 总reward
            j = 0                           # 步数
            while j < max_epLength:
                j += 1
                if np.random.rand(1) < sess.run(e) or sess.run(total_steps) < pre_train_steps:
                    a = np.random.randint(0, env.actions)
                else:
                    a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                s1, r, d = env.step(a)
                # env.render()
                s1 = processState(s1)
                sess.run(step_plus)
                # print(sess.run(total_steps))
                episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                if sess.run(total_steps) > pre_train_steps:
                    if sess.run(e) > endE:
                        sess.run(drop_e)               # 降低学习率
                    if sess.run(total_steps) % update_freq == 0:                                # 开始训练
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
            # print(rAll)
            myBuffer.add(episodeBuffer.buffer)
            rList.append(rAll)
            if i > 0 and i % 25 == 0:
                print('episode', i, ', average reward of last 25 episode', np.mean(rList[-25:]))
            if i > 0 and i % 1000 == 0:
                saver.save(sess, path + '/model.ckpt-' + str(i))
                print("Saved Model")
        saver.save(sess, path + '/model.ckpt-' + str(i))


if __name__ == '__main__':
    train()
#!/usr/bin/env python
from __future__ import print_function

from game import wrapped_flappy_bird as game
from collections import deque
import numpy as np
import random
import tensorflow as tf
import cv2.cv2 as cv2

GAME = 'bird'
ACTIONS = 2             # 可用操作数，flappy bird只有上（点）或不操作（不点）
GAMMA = 0.9            # decay rate of past observations
OBSERVE = 1000.  # 训练开始前随机探索的步数

EXPLORE = 3000000.      # 开始减小epsilon的步数
FINAL_EPSILON = 0.0001  # epsilon终值
INITIAL_EPSILON = 0.1  # epsilon初值
INITIAL_W = 0.05

REPLAY_MEMORY = 50000   # 记录池大小
BATCH = 32  # 每一batch样本数
REPLACE = 300

FRAME_PER_ACTION = 1    #
SP = 'D:/Codes/python/RL/MyFlappy/'       # 项目路径


# 创建卷积层
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

# 创建池化层


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 构建网络
def createNetwork(net_name):
    # 以net_name作为命名空间名
    with tf.variable_scope(net_name):
        # 建立参数集合c_names, 偏置和线性系数的初始化器
        c_names, w_initializer, b_initializer = \
            [net_name + '_params', tf.GraphKeys.GLOBAL_VARIABLES], \
            tf.random_normal_initializer(
                0., INITIAL_W), tf.constant_initializer(0.01)

        # 输入层
        s = tf.placeholder("float", [None, 80, 80, 4])

        # 第一层，80*80*4--->>>20*20*32--->>>10*10*32
        with tf.variable_scope('l1'):
            w1 = tf.get_variable(
                'w1', [8, 8, 4, 32], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable(
                'b1', [32], initializer=b_initializer, collections=c_names)
            conv1 = tf.nn.relu(conv2d(s, w1, 4) + b1)
            pool1 = max_pool_2x2(conv1)

        # 第二层，10*10*32--->>>5*5*64--->>>3*3*64
        with tf.variable_scope('l2'):
            w2 = tf.get_variable(
                'w2', [4, 4, 32, 64], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable(
                'b2', [64], initializer=b_initializer, collections=c_names)
            conv2 = tf.nn.relu(conv2d(pool1, w2, 2) + b2)
            pool2 = max_pool_2x2(conv2)

        # 第三层，3*3*64--->>>3*3*64--->>>576???
        with tf.variable_scope('l3'):
            w3 = tf.get_variable(
                'w3', [3, 3, 64, 64], initializer=w_initializer, collections=c_names)
            b3 = tf.get_variable(
                'b3', [64], initializer=b_initializer, collections=c_names)
            conv3 = tf.nn.relu(conv2d(pool2, w3, 1) + b3)
            pool3 = max_pool_2x2(conv2)
            pool3_flat = tf.reshape(pool3, [-1, 576])

        # 第四层,576--->>>512
        with tf.variable_scope('l4'):
            w4 = tf.get_variable(
                'w4', [576, 512], initializer=w_initializer, collections=c_names)
            b4 = tf.get_variable(
                'b4', [512], initializer=b_initializer, collections=c_names)
            fc1 = tf.nn.relu(tf.matmul(pool3_flat, w4) + b4)

        # 第五层，512--->>>2，输出q的预测值，节点数为所有action的总数
        with tf.variable_scope('l5'):
            w5 = tf.get_variable(
                'w5', [512, ACTIONS], initializer=w_initializer, collections=c_names)
            b5 = tf.get_variable(
                'b5', [ACTIONS], initializer=b_initializer, collections=c_names)
            fc2 = tf.matmul(fc1, w5) + b5

    return s, fc2, fc1


# 训练网络
def trainNetwork(s, q_pred, s_, q_next, sess):
    # 准备损失函数
    # a: 实际进行的操作，one-hot
    a = tf.placeholder("float", [None, ACTIONS])
    # readout_action：网络对该操作给出的预测
    readout_action = tf.reduce_mean(tf.multiply(q_pred, a), axis=1)
    # y: 公式给出的该操作的目标q值
    y = tf.placeholder("float", [None])
    # cost：总损失
    cost = tf.reduce_mean(tf.square(y - readout_action))
    # trainstep: 损失求导器
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # 准备参数更新器replace_target_op
    t_params = tf.get_collection(
        'next_q_net_params')  # 提取 target_net 的参数
    e_params = tf.get_collection(
        'pred_q_net_params')   # 提取  pred_net 的参数
    replace_target_op = [tf.assign(
        t, e) for t, e in zip(t_params, e_params)]  # 更新 target_net 参数

    # 启动游戏模拟器
    game_state = game.GameState()

    # 创建队列保存操作记录
    D = deque()

    # 进行第一次操作，得到D的初始数据
    # 设置操作为1
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    # 从模拟器获得输出图像（截屏）x，奖励r，结束状况terminal
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    # 利用opencv库将截屏格式化为80*80图像
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    # 将该图像复制4份作为初始状态s
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 初始化网络
    tf.summary.FileWriter(SP + "tensorboard/", sess.graph)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    # 开始训练
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy" != "angry":
        # 决定当步动作
        # 用当前状态s_t输入网络得到各action的预测q值
        readout_t = q_pred.eval(feed_dict={s: [s_t]})[0]
        # 用预测q值得到推荐的动作
        a_t = np.zeros([ACTIONS])
        action_index = 0
        # 对于一定epsilon的概率，随机选择一个动作，否则选择q值最大的动作
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # 在t > OBSERVE之前逐步减小epsilon，即逐步增大网络决定的次数
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 将获得的动作输入模拟器，得到输出图像x，奖励r，结束状况terminal
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        # 处理输出x，和之前三步的x一起得到新的状态s_t1
        x_t1 = cv2.cvtColor(cv2.resize(
            x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 将该步的{步前状态，动作，奖励，步后状态，是否停止}作为一条记录放入记录池
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 在探索阶段结束后，对神经网络进行训练
        if t > OBSERVE:
            # 从记录池中获取一批记录
            minibatch = random.sample(D, BATCH)
            # 分别获取其s,a,r,s_
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # 准备存储其目标值
            y_batch = []
            # 将其下一步输入网络，得到下一步所有action的q
            readout_j1_batch = sess.run(q_next, feed_dict={s_: s_j1_batch})
            # 对于每一条记录，用公式算出目标q（即y）
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:  # 碰到障碍物，终止
                    y_batch.append(r_batch[i])
                else:  # 即时奖励 + 下一阶段回报
                    y_batch.append(r_batch[i] + GAMMA *
                                   np.max(readout_j1_batch[i]))

            # 将y，a，s传入网络，进行训练
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        # 更新网络参数
        if t % REPLACE == 0:
            sess.run(replace_target_op)
            print("\ntarget_params_replaced")

        # 更新当前状态和时间步
        s_t = s_t1  # state 更新
        t += 1

        # 每10000次迭代保存当前网络
        # if t % 10000 == 0:
        #     saver.save(sess, SP + 'saved_networks/' +
        #                GAME + '-dqn', global_step=t)

        # 打印该步训练信息
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("terminal", terminal,
              "TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    s, q_pred, h_fc1_1 = createNetwork('pred_q_net')
    s_, q_next, h_fc1_2 = createNetwork('next_q_net')
    trainNetwork(s, q_pred, s_, q_next, sess)

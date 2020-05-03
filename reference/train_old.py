#!/usr/bin/env python
from __future__ import print_function

from game import wrapped_flappy_bird as game
from collections import deque
import numpy as np
import random
import tensorflow as tf
import cv2.cv2 as cv2
import sys
sys.path.append("game/")


'''
先观察一段时间（OBSERVE = 1000 不能过大），
获取state(连续的4帧) => 进入训练阶段（无上限）=> action

'''
GAME = 'bird'
ACTIONS = 2             # 可用操作数，flappy bird只有上（点）或不操作（不点）
GAMMA = 0.99            # decay rate of past observations
OBSERVE = 1000.         # 训练开始前随机探索的步数
EXPLORE = 3000000.      # 开始减小epsilon的步数
FINAL_EPSILON = 0.0001  # epsilon终值
INITIAL_EPSILON = 0.1   # epsilon初值
REPLAY_MEMORY = 50000   # 记录池大小
BATCH = 32              # 每一batch样本数
FRAME_PER_ACTION = 1    # ???
SP = 'D:/Codes/python/RL/DeepLearningFlappyBird/'       # 项目路径

# 创建权重变量及初始化


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# 创建偏置变量及其初始化


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


'''
padding = ‘SAME’=> new_height = new_width = W / S （结果向上取整）
padding = ‘VALID’=> new_height = new_width = (W – F + 1) / S （结果向上取整）
'''

# 创建卷积层


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

# 创建池化层


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


"""
 数据流：80 * 80 * 4  
 conv1(8 * 8 * 4 * 32, Stride = 4) + pool(Stride = 2)-> 10 * 10 * 32(height = width = 80/4 = 20/2 = 10)
 conv2(4 * 4 * 32 * 64, Stride = 2) -> 5 * 5 * 64 + pool(Stride = 2)-> 3 * 3 * 64
 conv3(3 * 3 * 64 * 64, Stride = 1) -> 3 * 3 * 64 = 576
 576 在定义h_conv3_flat变量大小时需要用到，以便进行FC全连接操作
"""

# 构建网络


def createNetwork():
    # 权重和偏差变量
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([576, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # 输入层
    s = tf.placeholder("float", [None, 80, 80, 4])

    # 隐层
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 576])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # 输出层（BATCH * 2）
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


# 训练网络
def trainNetwork(s, readout, h_fc1, sess):
    # 准备损失函数
    # a: 实际进行的操作，one-hot
    a = tf.placeholder("float", [None, ACTIONS])
    # readout_action：网络对该操作给出的预测
    readout_action = tf.reduce_mean(tf.multiply(readout, a), axis=1)
    # y: 公式给出的该操作的目标q值
    y = tf.placeholder("float", [None])
    # cost：总损失
    cost = tf.reduce_mean(tf.square(y - readout_action))
    # trainstep: 损失求导器
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # 启动游戏模拟器
    game_state = game.GameState()

    # 创建队列保存操作记录
    D = deque()

    # 文件记录
    # a_file = open(SP + "logs_" + GAME + "/readout.txt", 'w')
    # h_file = open(SP + "logs_" + GAME + "/hidden.txt", 'w')

    # 进行第一次操作，得到D的初始数据
    # 设置操作为1
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    # cv2.imwrite('x_t.jpg',x_t)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    tf.summary.FileWriter(SP + "tensorboard/", sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    """
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    """
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        # 预测结果（当前状态不同行为action的回报，其实也就 往上，往下 两种行为）
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            # 加入一些探索，比如探索一些相同回报下其他行为，可以提高模型的泛化能力。
            # 且epsilon是随着模型稳定趋势衰减的，也就是模型越稳定，探索次数越少。
            if random.random() <= epsilon:
                # 在ACTIONS范围内随机选取一个作为当前状态的即时行为
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                # 输出 奖励最大就是下一步的方向
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # scale down epsilon 模型稳定，减少探索次数。
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        # 先将尺寸设置成 80 * 80，然后转换为灰度图
        x_t1 = cv2.cvtColor(cv2.resize(
            x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        # x_t1 新得到图像，二值化 阈值：1
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        # 取之前状态的前3帧图片 + 当前得到的1帧图片
        # 每次输入都是4幅图像
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        # s_t: 当前状态（80 * 80 * 4）
        # a_t: 即将行为 （1 * 2）
        # r_t: 即时奖励
        # s_t1: 下一状态
        # terminal: 当前行动的结果（是否碰到障碍物 True => 是 False =>否）
        # 保存参数，队列方式，超出上限，抛出最左端的元素。
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # 获取batch = 32个保存的参数集
            minibatch = random.sample(D, BATCH)
            # get the batch variables
            # 获取j时刻batch(32)个状态state
            s_j_batch = [d[0] for d in minibatch]
            # 获取batch(32)个行动action
            a_batch = [d[1] for d in minibatch]
            # 获取保存的batch(32)个奖励reward
            r_batch = [d[2] for d in minibatch]
            # 获取保存的j + 1时刻的batch(32)个状态state
            s_j1_batch = [d[3] for d in minibatch]
            # readout_j1_batch =>(32, 2)
            y_batch = []
            readout_j1_batch = sess.run(readout, feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:  # 碰到障碍物，终止
                    y_batch.append(r_batch[i])
                else:  # 即时奖励 + 下一阶段回报
                    y_batch.append(r_batch[i] + GAMMA *
                                   np.max(readout_j1_batch[i]))
            # 根据cost -> 梯度 -> 反向传播 -> 更新参数
            # perform gradient step
            # 必须要3个参数，y, a, s 只是占位符，没有初始化
            # 在 train_step过程中，需要这3个参数作为变量传入

            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        # update the old values
        s_t = s_t1  # state 更新
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        # print info
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

        """     
        # write info to files
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        """


def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()

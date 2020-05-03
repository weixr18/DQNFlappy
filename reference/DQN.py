# 目标-老网络，用R和下一步Q最大值作为Q_目标
# 预测-新网络，直接输出Q_预测

import tensorflow as tf
import numpy as np
import cv2.cv2 as cv2


class DQN:
    # 所有初始化
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None,
        output_graph=False
    ):
        self.n_actions = n_actions      # 可能操作总数
        self.n_features = n_features    # 状态特征的维数
        self.lr = learning_rate         # 学习率
        self.gamma = reward_decay       # 下步q和奖励在目标q中的比值
        self.epsilon_max = e_greedy     # epsilon 的最大值
        self.replace_target_iter = replace_target_iter  # target_net 更新 的步数
        self.memory_size = memory_size  # 记忆池的上限
        self.batch_size = batch_size    # 每步训练抽取的记录条数
        self.epsilon_increment = e_greedy_increment  # epsilon 的增量
        # 是否开启探索模式, 并逐步减少探索次数
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0

        # 初始化全 0 记忆 [s, a, r, s_]
        # 和视频中不同, 因为 pandas 运算比较慢, 这里改为直接用 numpy
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        # 创建 [target_net, evaluate_net]
        self._build_net()

        # replace_target_op节点：用预测网络的参数更新目标网络的参数
        t_params = tf.get_collection(
            'target_net_params')  # 提取 target_net 的参数
        e_params = tf.get_collection(
            'pred_net_params')   # 提取  pred_net 的参数
        self.replace_target_op = [tf.assign(
            t, e) for t, e in zip(t_params, e_params)]  # 更新 target_net 参数

        # 建立tf会话
        self.sess = tf.Session()

        # 输出 tensorboard 文件
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        # 进行所有全局变量的初始化
        self.sess.run(tf.global_variables_initializer())

        # 记录所有 cost 变化, 用于最后 plot 出来观看
        self.cost_his = []

    # 构建网络计算图
    def _build_net(self):
        # -------------- 创建prediction网络，用于输出预测，参数实时学习 --------------
        # s:输入节点，即环境的当前状态。行数不定，列数为n_features(一个状态特征向量的维数)
        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name='s')

        # q_target节点:用公式算出的q目标值
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')

        # q_pred节点：prediction网络输出的q值
        with tf.variable_scope('pred_net'):
            # c_names?
            c_names, n_l1, w_initializer, b_initializer = \
                ['pred_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(
                    0., 0.3), tf.constant_initializer(0.1)

            # 第一层，输入维数（矩阵行数）为n_features
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # 第二层，输出q的预测值，节点数为所有action的总数
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_pred = tf.matmul(l1, w2) + b2

        # loss节点：用q_target和q_pred计算的损失函数
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_pred))

        # _train_op节点：使用RMSprop优化器更新参数？
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

        # -------------- 创建target网络，给出计算q_target所需的下一步的q值，参数滞后更新 --------------
        # s_节点：环境的下一状态，行数不定，列数为n_features(一个状态特征向量的维数)
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')

        # q_pred节点：target（滞后）网络输出的下步的q值
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params',
                       tf.GraphKeys.GLOBAL_VARIABLES]

            # 第一层，输入维数（矩阵行数）为n_features
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(
                    tf.matmul(self.s_, w1) + b1)

            # 第二层，输出q的预测值，节点数为所有action的总数
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 将记录更新到memory中
    def store_transition(self, s, a, r, s_):
        # 创建记录计数器
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 将本步状态s, 选择动作a, 奖励r, 下一步状态s_拼接为一条记录
        transition = np.hstack((s, [a, r], s_))

        # 在0-memory_size范围内循环更新memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 做出决策
    def choose_action(self, observation):
        # 增加一个维度
        observation = observation[np.newaxis, :]

        # 以一定概率让网络做决定，一定概率随机决定
        if np.random.uniform() < self.epsilon:
            # 将当前状态送入pred网络，得到每种action的推荐度，选出最大的
            actions_value = self.sess.run(
                self.q_pred, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # 学习过程，核心算法
    def learn(self):
        # 每replace_target_iter步将target网络的参数更新
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print("\ntarget_params_replaced")

        # 从记忆池中抽取batch_size条记录
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 对这批记录中的每一条，用滞后更新参数的target网络得到下一步的q，用实时更新参数的prediction网络得到s的q
        q_next, q_pred = self.sess.run(
            [self.q_next, self.q_pred],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features]
            }
        )

        # 对这一批记录中的每一条，计算其目标q

        # 先把q目标值统一设置为预测值。
        # 由于仅需（也仅能）更新q的{记录中选择了的那个action}的分量，
        # 因此其他分量不需更新，保持其与预测q相同即可
        q_target = q_pred.copy()
        # 生成一个0-batch_size - 1的数列备用
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 找到每条记录中选择了的那个action的编号
        pred_act_index = batch_memory[:, self.n_features].astype(int)
        # 找到每条记录中该action产生的reward
        reward = batch_memory[:, self.n_features + 1]

        # 核心公式，对于一批样本中的每一个记录中的s和选择了的a，其目标q值就是：
        # 选择这个a产生的reward，加上系数gamma乘以{这个a到达的下一个状态s_的所有可能动作的q}中的最大值
        q_target[batch_index, pred_act_index] = reward + \
            self.gamma * np.max(q_next, axis=1)

        # 将s传入预测网络得到q_pred，将q_target和q_pred传入loss节点计算损失，用损失训练预测网络
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        # 记录每一步的误差
        self.cost_his.append(self.cost)

        # 逐渐增加epsilon，直到给出的epsilon最大值
        self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon_max else self.epsilon_max
        # 学习步数加一
        self.learn_step_counter += 1


# 主函数
if __name__ == '__main__':
    DeepQNetwork = DQN(3, 4, output_graph=True)

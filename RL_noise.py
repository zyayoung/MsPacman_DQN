import numpy as np
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Lambda, Concatenate, Reshape, Conv2D, MaxPool2D, Activation, Flatten, GaussianNoise
from keras import backend as K
from keras.optimizers import RMSprop


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.008,
            reward_decay=0.9,
            e_greedy=0.9,
            e_greedy_start=0.0,
            memory_size=500,
            batch_size=32,
            replace_target_iter=300,
            e_greedy_increment=None
    ):

        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.42
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy_start if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            predict = self.evaluate_net.predict(s[np.newaxis, :])[0]
            #print(predict)
            action = predict.argmax()
        else:
            action = np.random.randint(self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.evaluate_net.get_weights())
            print("Target params replaced. epsilon:", self.epsilon)
            # self.evaluate_net.set_weights(n())

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.target_net.predict(batch_memory[:, -self.n_features:])
        q_eval = self.evaluate_net.predict(batch_memory[:, :self.n_features])
        q_eval4next = self.evaluate_net.predict(batch_memory[:, -self.n_features:])

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)
        selected_q_next = q_next[batch_index, max_act4next]

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        (self.evaluate_net.train_on_batch(batch_memory[:, :self.n_features], q_target))

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def _create_model(self):
        # model = Sequential()
        # model.add(Dense(12, input_shape=(self.n_features,), activation='relu'))
        # model.add(Dense(self.n_actions))
        # model.compile(RMSprop(self.lr), 'mse')
        # return model
        model_input = Input((self.n_features,))
        shared_output = Reshape((210, 160, 12))(model_input)
        shared_output = Conv2D(32, (8, 8), padding='same', strides=4)(shared_output)
        # shared_output = MaxPool2D((2, 2))(shared_output)
        shared_output = Activation('relu')(shared_output)
        shared_output = Conv2D(64, (4, 4), padding='same', strides=2)(shared_output)
        # shared_output = MaxPool2D((2, 2))(shared_output)
        shared_output = Activation('relu')(shared_output)
        shared_output = Conv2D(64, (3, 3), padding='same', strides=1)(shared_output)
        # shared_output = MaxPool2D((3, 3))(shared_output)
        shared_output = Activation('relu')(shared_output)
        shared_output = Flatten()(shared_output)
        # shared_output = Dense(1024, activation='elu')(shared_output)
        shared_output = Dense(512, activation='relu')(shared_output)
        # shared_output = Dense(256, activation='relu')(shared_output)
        # shared_output = GaussianNoise(1)(shared_output)
        A = Dense(self.n_actions)(shared_output)
        V = Dense(1)(shared_output)
        c = Concatenate()([V, A])
        model_output = Lambda(
            lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
        )(c)
        model = Model(model_input, model_output)
        model.compile(RMSprop(self.lr), 'mse')
        # model.compile('adam', 'mse')
        # model.summary()
        return model

    def _build_net(self):
        self.evaluate_net = self._create_model()
        self.target_net = self._create_model()
        # self.target_net.summary()

    def save(self):
        self.target_net.save('./models/target_net.h5')

    def load(self):
        self.evaluate_net = load_model('./models/target_net.h5')
        self.target_net = load_model('./models/target_net.h5')
        self.target_net.compile(RMSprop(self.lr), 'mse')
        self.evaluate_net.compile(RMSprop(self.lr), 'mse')


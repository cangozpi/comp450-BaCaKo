import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from networks import ActorCriticNetwork
import numpy as np


class Agent:
    def __init__(self, epsilon=0.2, alpha=0.0003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        self.epsilon = epsilon
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha, clipnorm=1.0, clipvalue=0.5))


        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_state_memory = []
        self.done_memory = []

    def choose_action(self, observation, epsilon=0.2):

        state = tf.convert_to_tensor([observation])

        _, actionEstimation = self.actor_critic(state)

        actionEstimationNp = actionEstimation.numpy()

        returnArray = []
        randChance = np.random.rand()

        if self.epsilon > randChance:  # epsilon prob any action
            randActionSelection = np.random.rand()
            if randActionSelection < 0.33:  # buyi

                returnArray.append(actionEstimationNp[0][0])
            elif randActionSelection < 0.66:  # hold

                returnArray.append(actionEstimationNp[0][1])
            else:  # sell

                returnArray.append(actionEstimationNp[0][2])
        else:  # 1-epsilon prob take max action

            buyAmount = actionEstimationNp[0][0]
            hold = actionEstimationNp[0][1]
            sellAmount = actionEstimationNp[0][2]
            maxAction = max(buyAmount, hold, sellAmount)
            if maxAction == buyAmount:

                returnArray.append(actionEstimationNp[0][0])
            elif maxAction == hold:  # hold

                returnArray.append(actionEstimationNp[0][1])
            elif maxAction == sellAmount:

                returnArray.append(actionEstimationNp[0][2])

        returnArray.append(actionEstimationNp[0][0])
        returnArray.append(actionEstimationNp[0][1])
        returnArray.append(actionEstimationNp[0][2])
        action = returnArray[0]

        self.action = action

        return returnArray


    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self):
        actions = self.action_memory
        states = self.state_memory
        state_s = self.next_state_memory
        rewards = self.reward_memory
        dones = self.done_memory


        tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            actor_loss = 0
            critic_loss = 0
            total_loss = 0
            count = 0
            for i, (action, reward, done, state, state_) in enumerate(zip(actions, rewards, dones, states, state_s)):
                count += 1
                state = tf.convert_to_tensor([state[1:]], dtype=tf.float32)
                state_ = tf.convert_to_tensor([state_[1:]], dtype=tf.float32)

                state_value, _ = self.actor_critic(state)
                state_value_, _ = self.actor_critic(state_)

                state_value = tf.squeeze(state_value)
                state_value_ = tf.squeeze(state_value_)

                action_probs = tfp.distributions.Categorical(probs=action[1:])

                log_prob = action_probs.log_prob(action[0])

                delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
                if log_prob != 0 and log_prob != float('-inf'):
                    actor_loss += ((-log_prob * delta) - actor_loss) / count
                else:
                    pass
                critic_loss += ((delta ** 2) - critic_loss) / count
                total_loss += ((actor_loss + critic_loss) - total_loss) / count

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)

        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))
        self.resetMemory()

    def resetMemory(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_state_memory = []
        self.done_memory = []


    def store_transition(self, curState, action, reward, nextState, isDone):
        self.action_memory.append(action)
        self.state_memory.append(curState)
        self.reward_memory.append(reward)
        self.next_state_memory.append(nextState)
        self.done_memory.append(isDone)

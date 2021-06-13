import numpy as np
import matplotlib.pyplot as plt
from actor_critic import Agent
from Environment import Env
import tensorflow as tf
#from utils import plot_learning_curve

def plotData(dataToPlot,xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(dataToPlot)
    plt.show()

if __name__ == '__main__':
  #  tf.debugging.set_log_device_placement(True)
    env = Env()
    stAgent = Agent(epsilon = 0.2,alpha=1e-7, n_actions=3)
    n_episodes = 1000
    filename = 'bacako.png'
    type = 0 #no intrinsic short term
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    avg_score_history = []
    load_checkpoint = False

    if load_checkpoint:
        stAgent.load_models()

    for i in range(n_episodes):

        observation = env.startState(type)
        observation_ = []
        reward = 0
        done = False
        score = 0
        while not done:
            action = stAgent.choose_action(observation[1:])

            observation_, reward, done = env.step(observation, action, type)
            stAgent.store_transition(observation, action, reward, observation_, done)
            score += reward
            # if not load_checkpoint:
            #stAgent.learn(observation[1:], reward, observation_[1:], done)
            observation = observation_

        score_history.append(score)
        stAgent.learn()
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)


        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                stAgent.save_models()

        print('episode ' + str(i) + ' score: ' + str(score) + ' avg_score' + str(avg_score))
    plotData(score_history, "time", "score")
    plotData(avg_score_history, "time", "avg_score")
    #if not load_checkpoint:
     #   x = [i+1 for i in range(n_episodes)]
      #  plot_learning_curve(x, score_history, figure_file)


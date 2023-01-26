import numpy as np
import math


def SARSA_update_rule(Q_func, transition, alpha, gamma):
    state, action, reward, next_state, next_action, done = transition

    Q_func[state, action] += alpha * (reward + gamma * Q_func[next_state, next_action] * (1 - done) - Q_func[state, action])


    return Q_func


def Q_update_rule(Q_func, transition, alpha, gamma):
    state, action, reward, next_state, next_action, done = transition

    Q_func[state, action] += alpha * (reward + gamma * np.max(Q_func[next_state]) * (1 - done) - Q_func[state, action])


    return Q_func

class LT_agent():

    def __init__(self, n_states, n_actions, gamma, alpha):

        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.Q_func = np.zeros((n_states, n_actions))


    def policy(self, state, explore_rate):
        '''
        chooses an action based on the agents value function and the current explore rate
        :param state: the current state given by the environment
        :param explore_rate: the chance of taking a random action
        :return: the action to be taken
        '''


        if np.random.random() < explore_rate:
            action = np.random.choice(range(self.n_actions))


        else:
            action = np.argmax(self.Q_func[state])

        return action

    def get_explore_rate(self, episode, decay, min_r = 0, max_r = 1):
        '''
        Calculates the logarithmically decreasing explore or learning rate

        Parameters:
            episode: the current episode
            MIN_LEARNING_RATE: the minimum possible step size
            MAX_LEARNING_RATE: maximum step size
            denominator: controls the rate of decay of the step size
        Returns:
            step_size: the Q-learning step size
        '''

        # input validation
        if not 0 <= min_r <= 1:
            raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 <= max_r <= 1:
            raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 < decay:
            raise ValueError("decay needs to be above 0")

        rate = max(min_r, min(max_r, 1.0 - math.log10((episode + 1) / decay)))

        return rate


class TD_agent(LT_agent):

    def __init__(self, n_states, n_actions, gamma = 0.95, alpha = 0.01, update_rule=None):
        LT_agent.__init__(self, n_states, n_actions, gamma, alpha)

        if update_rule is None:
            self.update_rule = Q_update_rule
        else:
            self.update_rule = update_rule

    def update_Q(self,transition):
        '''
        updates the agents value function based on the experience in transition
        :param transition:
        :return:
        '''


        self.Q_func = self.update_rule(self.Q_func, transition, self.alpha, self.gamma)



class MC_agent(LT_agent):
    def __init__(self, n_states, n_actions, gamma = 0.95, alpha = 0.01):


        LT_agent.__init__(self, n_states, n_actions, gamma, alpha)

        self.visit_count = np.zeros((n_states, n_actions))
        self.return_sum = np.zeros((n_states, n_actions))


    def update_Q(self, episode):
        ret = 0

        for transition in episode[::-1]:
            state, action, reward, next_state, next_action, done = transition
            ret = self.gamma*ret + reward

            self.visit_count[state, action] += 1
            self.return_sum[state, action] += ret
            self.Q_func[state, action] = self.return_sum[state, action]/self.visit_count[state, action]












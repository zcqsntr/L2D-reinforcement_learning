import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import time
def plot_value(value_func, grid_shape, p_map):
    fig, axs = plt.subplots(1,2, figsize = [15,10])
    axs[0].set_title('State-action value')
    im = axs[0].imshow(value_func.T)
    axs[0].set_xlabel('State')
    axs[0].set_ylabel('Action')
    cbar = fig.colorbar(im, ax=axs[0])
    cbar.set_label('State-action value')


    axs[1].set_title('State value and policy')
    state_value = np.max(value_func, axis=1).reshape(grid_shape)
    policy = np.array([p_map[np.argmax(value_func[i])] for i in range(len(value_func))]).reshape(grid_shape)

    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):
            axs[1].text(col, row, policy[row, col], color='red')

    im = axs[1].imshow(state_value)
    cbar = fig.colorbar(im, ax=axs[1])
    cbar.set_label('State value')


def plot_returns(returns):
    plt.figure()
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')


def plot_explore(explore_func, n_episodes):
    rates = [explore_func(episode, n_episodes / 11) for episode in range(n_episodes)]
    plt.figure()
    plt.plot(rates)
    plt.xlabel('Episode')
    plt.ylabel('Explore rate')
    
    
def train_agent(agent, env,  n_episodes = 10001, monte_carlo = False, display=False, simple_text = True):

    returns = []
    
    if display:
      display = Display(visible=0, size=(400, 300))
      display.start()
    
    for episode in range(0,n_episodes): # for each episode

        e_return = 0 # sum the reward we get this episode
        e_transitions = [] # memory of all tranisitions seen in this episode
        done = False # has the episode finished?
        
        state, prob = env.reset() # reset env to initial state

        explore_rate = agent.get_explore_rate(episode, n_episodes/11)
        
        action = agent.policy(state, explore_rate) 

        if display: # display the current agent-evironment state
            #env.render()
            print(explore_rate)
            plt.imshow(env.render())
            time.sleep(0.1)
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())
        
        
        while not done: # run the episode until a terminal state reached
 
            next_state, reward, done, info, prob = env.step(action) # take an action and get the resulting state from the env
          
            next_action = agent.policy(next_state, explore_rate) # get the next action to apply from agent's policy
            transition = (state, action, reward, next_state, next_action, done) # create the SARSA transition
            e_transitions.append(transition) # add to the memory

            if not monte_carlo: # update the Q function of temporal difference agents
                agent.update_Q(transition)

            state = next_state
            action = next_action

            e_return += reward
            
            if display: # display the current agent-evironment state
                #print(env.render())
                plt.imshow(env.render())
                time.sleep(0.1)
                ipythondisplay.clear_output(wait=True)
                ipythondisplay.display(plt.gcf())
                
            if done:
                break


        returns.append(e_return)
        
        if monte_carlo: # update the q function of a monte carlo agent
            agent.update_Q(e_transitions)

        if episode % 100 == 0: # print results of current episode
            print('episode:', episode, ', explore_rate:', explore_rate, ', return:', e_return)
            
    if display:
      ipythondisplay.clear_output(wait=True) 
            
    return returns
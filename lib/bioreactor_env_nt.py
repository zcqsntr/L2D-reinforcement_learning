import numpy as np
from scipy.integrate import odeint

def target_reward(x, targ):
    '''
    caluclates the reward based on the distance x is from the target
    :param x:
    :param targ:
    :return:
    '''
    N = x[0]
    SE = np.abs(x-targ)
    reward = (1 - sum(SE/targ)/2)/10
    done = False
    if N < 1000:
        reward = - 1
        done = True

    return reward, done

def monod(C, C0, umax, Km, Km0):
    '''
    Calculates the growth rate based on the monod equation

    Parameters:
        C: the concetrations of the auxotrophic nutrients for each bacterial
            population
        C0: concentration of the common carbon source
        Rmax: array of the maximum growth rates for each bacteria
        Km: array of the saturation constants for each auxotrophic nutrient
        Km0: array of the saturation constant for the common carbon source for
            each bacterial species
    '''

    # convert to numpy

    growth_rate = ((umax * C) / (Km + C)) * (C0 / (Km0 + C0))

    return growth_rate

def xdot_control(x, t, u):
    '''
    Calculates and returns derivatives for the numerical solver odeint

    Parameters:
        S: current state
        t: current time
        Cin: array of the concentrations of the auxotrophic nutrients and the
            common carbon source
        params: list parameters for all the exquations
        num_species: the number of bacterial populations
    Returns:
        xdot: array of the derivatives for all state variables
    '''
    q = 0.5
    y, y0, umax, Km, Km0 = 520000, 480000, 1, 0.00048776, 0.00006845928

    # extract variables

    N, C, C0 = x
    R = monod(C, C0, umax, Km, Km0)

    # calculate derivatives
    dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (u[0] - C) - (1 / y) * R * N
    dC0 = q * (1. - C0) - 1 / y0 * R * N

    # consstruct derivative vector for odeint
    xdot = [dN, dC, dC0]

    return xdot
    
    
def xdot_product(x, t, u):
    '''
    Calculates and returns derivatives for the numerical solver odeint

    Parameters:
        S: current state
        t: current time
        Cin: array of the concentrations of the auxotrophic nutrients and the
            common carbon source
        params: list parameters for all the exquations
        num_species: the number of bacterial populations
    Returns:
        xdot: array of the derivatives for all state variables
    '''
    q = 0.5
    y, y0, umax, Km, Km0 = [np.array(x) for x in [[480000., 480000.], [520000., 520000.], [1., 1.1], [0.00048776, 0.000000102115], [0.00006845928, 0.00006845928]]]

    # extract variables
    N = x[:2]
    C = x[2:4]
    C0 = x[4]
    A = x[5]
    B = x[6]
    P = x[7]

    R = monod(C, C0, umax, Km, Km0)

    # calculate derivatives
    dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (u - C) - (1 / y) * R * N
    dC0 = q*(0.1 - C0) - sum(1/y0[i]*R[i]*N[i] for i in range(2))

    dA = N[0] - 2 * A ** 2 * B - q * A
    dB = N[1] - A ** 2 * B - q * B
    dP = A ** 2 * B - q * P

    # consstruct derivative vector for odeint
    xdot = np.append(dN, dC)
    xdot = np.append(xdot, dC0)
    xdot = np.append(xdot, dA)
    xdot = np.append(xdot, dB)
    xdot = np.append(xdot, dP)


    return xdot


class BioreactorEnv():

    '''
    Chemostat environment that can handle an arbitrary number of bacterial strains where all are being controlled

    '''

    def __init__(self, xdot, reward_func, sampling_time, num_controlled_species, initial_x, max_t, n_states = 10, n_actions = 2, continuous_s = False):
        '''
        Parameters:
            param_file: path of a yaml file contining system parameters
            reward_func: python function used to coaculate reward: reward = reward_func(state, action, next_state)
            sampling_time: time between sampl-and-hold intervals
            scaling: population scaling to prevent neural network instability in agent, aim to have pops between 0 and 1. env returns populations/scaling to agent
        '''
        one_min = 0.016666666667
        
        self.scaling = 1

        self.xdot = xdot
        self.xs = []
        self.us = []
        self.sampling_time = sampling_time*one_min
        self.reward_func = reward_func

        self.u_bounds = [0,0.1]
        self.N_bounds = [0, 50000]

        self.u_disc = n_actions
        self.N_disc = n_states
        self.num_controlled_species = num_controlled_species
        self.initial_x = initial_x
        self.max_t = max_t
        self.continuous_s = continuous_s

    def step(self, action):
        '''
        Performs one sampling and hold interval using the action provided by a reinforcment leraning agent

        Parameters:
            action: action chosen by agent
        Returns:
            state: scaled state to be observed by agent
            reward: reward obtained buring this sample-and-hold interval
            done: boolean value indicating whether the environment has reached a terminal state
        '''
        u = self.action_to_u(action)

        #add noise
        #Cin = np.random.normal(Cin, 0.1*Cin) #10% pump noise

        self.us.append(u)

        ts = [0, self.sampling_time]


        sol = odeint(self.xdot, self.xs[-1], ts, args=(u,))[1:]

        self.xs.append(sol[-1,:])

        self.state = self.get_state()

        reward, done = self.reward_func(self.xs[-1])
        
        if len(self.xs) == self.max_t:
            done = True

        return self.state, reward, done, None, 1

    def get_state(self):
        '''
        Gets the state (scaled bacterial populations) to be observed by the agent

        Returns:
            scaled bacterial populations
        '''
        if not self.continuous_s:
            return self.pop_to_state(self.xs[-1][0:self.num_controlled_species])
        else:
            return self.xs[-1][0:self.num_controlled_species]/100000

    def action_to_u(self,action):
        '''
        Takes a discrete action index and returns the corresponding continuous state
        vector

        Paremeters:
            action: the descrete action
            num_species: the number of bacterial populations
            num_Cin_states: the number of action states the agent can choose from
                for each species
            Cin_bounds: list of the upper and lower bounds of the Cin states that
                can be chosen
        Returns:
            state: the continuous Cin concentrations correspoding to the chosen
                action
        '''

        # calculate which bucket each eaction belongs in
        buckets = np.unravel_index(action, [self.u_disc] * self.num_controlled_species)

        # convert each bucket to a continuous state variable
        u = []
        for r in buckets:
            u.append(self.u_bounds[0] + r*(self.u_bounds[1]-self.u_bounds[0])/(self.u_disc-1))

        u = np.array(u).reshape(self.num_controlled_species,)

        return np.clip(u, self.u_bounds[0], self.u_bounds[1])

    def pop_to_state(self, N):
        '''
        discritises the population of bacteria to a state suitable for the agent
        :param N: population
        :return: discitised population
        '''
        step = (self.N_bounds[1] - self.N_bounds[0])/self.N_disc

        N = np.clip(N, self.N_bounds[0], self.N_bounds[1]-1)


        return np.ravel_multi_index((N//step).astype(np.int32), [self.N_disc]*self.num_controlled_species)

    def reset(self, initial_x = None):
        '''
        Resets env to inital state:

        Parameters:
            initial_S (optional) the initial state to be reset to if different to the default
        Returns:
            The state to be observed by the agent
        '''
        
        if initial_x is None:
            initial_x = self.initial_x

        self.xs = [initial_x]
        self.us = []
        return (self.get_state(),1)
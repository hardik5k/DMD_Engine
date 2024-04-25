# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 12 # number of cities, ranges from 0 ..... m-1
t = 24 # number of hours, ranges from 0 .... t-1
M = 10
FC = 0.95


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        ## Retain (0,0) state and all states of the form (i,j) where i!=j
        self.action_space= [(pickup_location, drop_location) for pickup_location in range(m) for drop_location in range(m) if pickup_location!=drop_location]
        self.action_space.insert(0, (-1,-1))         # wait  
        self.action_space.insert(0, (-1,-2))          # accept requests
        ## All possible combinations of (m,t,d) in state_space
        self.state_space = [(driver_location, current_time, status) for driver_location in range(m) for current_time in range(1, t + 1) for status in range(2)]   
        # status = 0 ---> jobless
        # status = 1 ---> accepting requests

        self.state_init = self.state_space[np.random.choice(len(self.state_space))] 

        # Start the first round
        self.reset()
        


    ## Encoding state for NN input
    ## NOTE: Considering Architecture 2 given in the problem statement (where Input: STATE ONLY)
    ## --->Used in Agent_Architecture2_(Input_State_only).ipynb

    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint:
        The vector is of size m + t + d."""
        curr_loc, curr_time, curr_status = state

        ## Initialize arrays
        loc_arr = np.zeros(m, dtype=int)   # For location
        time_arr= np.zeros(t + 1, dtype=int)   # For time
        status_arr= np.zeros(2, dtype= int)   # For day

        ## Encoding respective arrays
        loc_arr[curr_loc] = 1
        time_arr[curr_time] = 1
        status_arr[curr_status] = 1

        ## Horizontal stacking to get vector of size m+t+d
        state_encod = np.hstack((loc_arr, time_arr, status_arr))
        state_encod= state_encod.tolist()
       
        return state_encod

    ## Encoding (state-action) for NN input
    ## Use this function if you are using architecture-1 
    ## def state_encod_arch2(self, state, action):
    ##     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into
    ## a vector format. Hint: The vector is of size m + t + d + m + m."""


    
    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        city_requests_matrix = np.load('city_requests_matrix.npy')
        requests = city_requests_matrix[state[0]][state[1]]
        
        ## (0,0) implies no action. The driver is free to refuse customer request at any point in time.
        ## Hence, add the index of  action (0,0)->[0] to account for no ride action.
        if (state[2] == 0):
            possible_actions_index = [0, 1]
        else:
            possible_actions_index = random.sample(range(2, (m - 1) * m + 1), requests) + [1]  
       
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index, actions   


    def update_time_day(self, curr_time, ride_duration):
        """
        Takes in the current time, current day and duration taken for driver's journey and returns
        updated time and updated day post that journey.
        """
        ride_duration = int(ride_duration)
        next_time = (curr_time + ride_duration) % 24
        if next_time == 0:
            return 24
        else:
            return next_time

    def get_ride_duration(self, curr_time, distance):
        speed_matrix = np.load('speed_matrix.npy')
        time = 0
        while (distance > 0):
            if (distance <= speed_matrix[curr_time]):
                return time + 1
            else:
                distance -= speed_matrix[curr_time]
                time += 1
        print("TIME CALC", time, distance)
        return time


    def get_next_state_and_time_func(self, state, action):
        distance_matrix = np.load('distance_matrix.npy')
        """Takes state, action and Time_matrix as input and returns next state, wait_time, transit_time, ride_time."""
        next_state = []
        
        # Initialize various times
        total_distance   = 0
        transit_distance = 0         # To go from current location to pickup location
        ride_distance    = 0   
        total_duration = 0      # From Pick-up to drop
        
        # Derive the current location, time, day and request locations
        curr_loc, curr_time, curr_status = state
        pickup_loc, drop_loc = action
        

        if (curr_status == 0): 
            if (drop_loc == -1):
                next_state = [curr_loc, self.update_time_day(curr_time, 1), 0]
                total_duration = 1
            elif (drop_loc == -2):
                next_state = [curr_loc, curr_time, 1]
        elif (curr_status == 1):
            if (drop_loc == -1):
                next_state = [curr_loc, self.update_time_day(curr_time, 1), 0]
                total_duration = 1
            else:
                if (curr_loc != pickup_loc):
                    transit_distance = distance_matrix[curr_loc][pickup_loc]
                ride_distance = distance_matrix[pickup_loc][drop_loc]
                total_distance = transit_distance + ride_distance
                print(curr_loc, pickup_loc, drop_loc, total_distance, ride_distance, transit_distance)
                total_duration = self.get_ride_duration(curr_time, total_distance)

                next_state = [drop_loc, self.update_time_day(curr_time, total_duration), 1]
        
        return next_state, total_distance, total_duration
        
    def next_state_func(self, state, action):
        """Takes state, action and Time_matrix as input and returns next state"""
        next_state= self.get_next_state_and_time_func(state, action)[0]   ## get_next_state_and_time_func() defined above       
        return next_state
    
    def wait_probabilistic_function(self):
        if np.random.rand() < 0.3:
            return 4
        else:
            return 0
        
    
    def reward_func(self, state, action):
        """Takes in state, action and Time_matrix and returns the reward"""
        ## get_next_state_and_time_func() defined above 
        price_matrix = np.load('price_matrix.npy')
        curr_loc, curr_time, curr_status = state
        pickup_loc, drop_loc = action
        next_state, total_distance, total_duration = self.get_next_state_and_time_func(state, action)

        if (curr_status == 0): 
            if (drop_loc == -1):
                reward = self.wait_probabilistic_function()
            elif (drop_loc == -2):
                reward = 0.5
        elif (curr_status == 1):
            if (drop_loc == -1):
                reward = self.wait_probabilistic_function()
            else:
                reward = price_matrix[pickup_loc][drop_loc][curr_time] - FC * ((total_distance) / M)


                
        
        return reward
    
    def step(self, state, action):
        """
        Take a trip as a cab driver. Takes state, action and Time_matrix as input and returns next_state, reward and total time spent
        """
        # Get the next state and the various time durations
        next_state, total_distance, total_duration, = self.get_next_state_and_time_func(state, action)

        # Calculate the reward and total_time of the step
        reward = self.reward_func(state, action)

        
        return next_state, reward, total_duration

    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
if (__name__ == '__main__'):
    m = 12
    t = 24
    env = CabDriver()
    action_space, state_space, state = env.reset()
    initial_state = state
    episode_time = 2
    total_time = 0
    done = False
    def get_action( state):
        possible_actions_index, actions= env.requests(state)
        return random.choice(possible_actions_index)
    
    while done!= True:
        # 1. Pick epsilon-greedy action from possible actions for the current state.
        action= get_action(state)
        
        # 2. Evaluate your reward and next state
        next_state, reward, step_time = env.step(state, env.action_space[action])
        
        # 3. Total time driver rode in this episode
        total_time += step_time
        if (total_time > episode_time):
            done = True
        else:
            # 4. Append the experience to the memory
            print(state, env.action_space[action], reward, next_state, done)   ## Note: Here action is action index
            state = next_state



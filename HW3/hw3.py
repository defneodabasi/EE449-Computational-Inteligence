# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:59:21 2024

@author: defne
"""

import utils
import numpy as np
from matplotlib import pyplot as plt
#from animation import animation_gif used to capture the movement of the agent along the maze, then commented
import cv2
import copy

class MazeEnvironment:
    
    def __init__(self):
        #Define the maze layout, rewards, action space (up, down, left, right)
        
        self. maze = maze
        self.start_pos = (0,0) # Start position of the agent
        self.current_pos = self.start_pos
        
        self.state_penalty = -1
        self.trap_penalty = -100
        self.goal_reward = 100
        
        self.actions = {0: (-1, 0), #Move up
                        1: (1, 0),  #Move down
                        2: (0,- 1), #Move left
                        3: (0, 1)   #Move right
                        }  
        
    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        '''
        Moving the agent according to an action.
        Due to the probabilistic nature of the environment the agent may move different than the chosen direction
        Probabilities:
            
        * Probability of going for the chosen direction: 0.75
        * Probability of going opposite of the chosen direction: 0.05
        * Probability of going each of perpendicular routes of the chosen direction: 0.10
        
        '''
        done = False
        reward = 0

        # Defining the possible moves       
        action = self.actions[action] # Chosen action
        
        opposite = (-action[0], -action[1])
        perp_right = (action[1], -action[0])
        perp_left = (-action[1],  action[0])
        
        moves = [action, opposite, perp_right, perp_left]
        probabilities = [0.75, 0.05, 0.10, 0.10] # chosen , opposite, perpendicular respectively.
        
        chosen_action = moves[np.random.choice(len(moves), p=probabilities)]
        
        #New position is calculated
        new_position = (self.current_pos[0] + chosen_action[0], self.current_pos[1] + chosen_action[1])
        
        # Check boundaries and whether the new position is an obstacle
        if (0 <= new_position[0] < self.maze.shape[0] and  #Horizatonal boundaries
            0 <= new_position[1] < self.maze.shape[1] and  #Vertical boundaries
            self.maze[new_position[0], new_position[1]] != 1): #Not an obstacle
            
            self.current_pos = new_position
        #Next position is on the boundaries
        else:
            reward += -5 #for discouraging the agent to stay at its current state due to hitting the boundaries
    
        # If new position is out of bounds or an obstacle, stay in current position
        #current position is maintained
        
        #Reward is calculated
        cell = self.maze[self.current_pos[0], self.current_pos[1]]    
        if cell == 0:
            reward += self.state_penalty
        # cell == 1 is a boundary
        if cell == 2:
            reward += self.trap_penalty
            done = True
        if cell == 3:
            reward += self.goal_reward
            done = True
        # Boundary or obstacle control
        return self.current_pos , reward, done
    
class MazeTD0(MazeEnvironment): #Inherited from MazeEnvironment
    def __init__(self, maze, alpha = 0.1, gamma = 0.95, epsilon = 0.2, episodes = 10000):
        super().__init__()
        
        self.maze = maze
        self.alpha = alpha #Learning rate
        self.gamma = gamma #Discount factor
        self.epsilon = epsilon #Exploration rate
        self.episodes = episodes
        self.utility =  np.zeros(self.maze.shape) #Encourage exploration
        
        # self.utility[maze == 2] = -1000 #trap
        # self.utility[maze == 3] = 1000 #goal
        self.utility[maze == 1] = -1000 #boundary
        
        self.convergence_history = []
        
    def choose_action(self, state):
        #Explore and Exploit: Choose the best action based on current utility values
        # Discourage invalid moves
        
        #Exploration:
        if np.random.rand() < self.epsilon: #any action
            return np.random.choice(list(self.actions.keys()))
        
        #Exploitation:
        else:
            #Geting the utilities of all possible actions from current state
            possible_actions = {}
            #Invalid moves are not considered in the exploitation
            for action, move in self.actions.items():
                next_pos = (state[0] + move[0], state[1] + move[1])
                if 0 <= next_pos[0] < self.maze.shape[0] and 0 <= next_pos[1] < self.maze.shape[1]:
                    possible_actions[action] = self.utility[next_pos]
                    
            #Choose the action with the highest utility
            return max(possible_actions, key = possible_actions.get) if possible_actions else None
    
    def update_utility_value(self, current_state, reward, new_state):
        current_value = self.utility[current_state]
        
        new_value = self.utility[new_state]
            
        new_value = current_value + self.alpha * ((reward + self.gamma * new_value) - current_value)
        
        #TD(0) update formula
        self.utility[current_state] = new_value
    
    def run_episodes(self):
        
        selected_episodes = [0, 49, 99, 999, 4999, 9999]
        utility_filenames = []
        policy_filenames = []
        mask = (maze != 2) & (maze != 3)

        # starting_episode = 200
        # ending_episode = 201
        # i = 0
        
        for episode in range(self.episodes):
            print("Episode:",episode)
            state = self.reset()
            
            prev_utility = np.copy(self.utility) #before the update
            
            done = False
            while not done:
                # if (starting_episode <= episode <= ending_episode):
                    # i += 1
                    # plt.clf()
                    # plt.imshow(self.maze, cmap='hot', interpolation='nearest')
                    # plt.scatter(self.current_pos[1], self.current_pos[0], c='blue', s=100)  # Agent's position
                    # plt.title(f'Maze_{episode}')
                    # if i<=9:
                    #     plt.savefig(f"C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots\\figure_000{i}")   # save the figure to file
                    # elif 10 <= i <= 99:
                    #     plt.savefig(f"C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots\\figure_00{i}")   # save the figure to file
                    # elif 100 <= i <= 999:
                    #     plt.savefig(f"C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots\\figure_0{i}")   # save the figure to file
                    # else:
                    #     plt.savefig(f"C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots\\figure_{i}")   # save the figure to file
                
                state = self.current_pos
                action = self.choose_action(state)
                next_state, reward, done = self.step(action)
                
                self.update_utility_value(state, reward, next_state)
            
            
            # Calculate the sum of absolute differences from the previous utility function
            diff = np.sum(np.abs(np.copy(self.utility[mask]) - prev_utility[mask]))
            self.convergence_history.append(diff)
                
            # if episode in selected_episodes:
                
            #     self.utility[maze == 2] = -1000
            #     self.utility[maze == 3] = 1000
                
            #     ax = utils_update.plot_value_function(self.utility, self.maze, episode)
            #     filename_1 = f'Utility_Episode_{episode + 1}_alpha_0_001.png'
            #     ax.figure.savefig(filename_1)
            #     utility_filenames.append(filename_1)
            #     plt.close()
                
            #     ax = utils_update.plot_policy(self.utility, self.maze, episode)
            #     filename_2 = f'Policy_Episode_{episode + 1}_alpha_0_001.png'
            #     ax.figure.savefig(filename_2)
            #     policy_filenames.append(filename_2)
            #     plt.close()
                
            #     self.utility[maze == 2] = 0
            #     self.utility[maze == 3] = 0
                    
        plt.show()        
        return self.utility, utility_filenames, policy_filenames

def plot_episodes(images, parameters, utility_or_policy):
    num_rows = 2
    num_cols = 3
    #plt.figure(fig_size=(24,12))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(40,20)) # Adjust the figsize based on your requirement
    
    for i, img_path in enumerate(images):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        row = i // num_cols  # Calculate row index
        col = i % num_cols   # Calculate column index
        axs[row, col].imshow(img)
        #axs[row, col].set_title(f'Episode: {episodes[i]}') # Title with generation
        axs[row, col].axis('off') # Hide axis
        
    plt.suptitle(f"{utility_or_policy} values with Parameters: {parameters}", fontsize = 30) # Main title with parameter
    plt.tight_layout()
    plt.show()
    
    return fig

def exponential_moving_average(history, alpha):
    ema = [history[0]]
    for x in history[1:]:
        ema.append(alpha * x + (1-alpha) * ema[-1])
    return ema 

#maze layout
maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 3],
    [0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
])

alpha = 1
gamma = 0.95
epsilon = 0.2 
episodes = 10000

maze_td0 = MazeTD0(maze, alpha, gamma, epsilon, episodes)
final_values, utility_image, policy_image = maze_td0.run_episodes()

history = maze_td0.convergence_history
#history_smoothed = moving_average(history, 50)
history_smoothed = exponential_moving_average(history, 0.02)

parameter = 'alpha = 1'
plt.figure(figsize=(20, 10))
episodes = np.arange(1,10001) # Episodes from 1 to 10000
plt.plot(episodes,history_smoothed)
plt.xlabel('Episode')
plt.ylabel('Sum of Absolute Differences')
plt.title(f'Convergence of Value Function for {parameter}')
plt.grid(True)
plt.savefig(f'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\TD_learning\\alpha\\alpha_1\\Convergence_for_parameter_{parameter}.png')
plt.show()

fig1 = plot_episodes(utility_image, parameter, 'Utility')
fig1.savefig(f'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\TD_learning\\alpha\\alpha_1\\Episodes_utility_{parameter}.png')

fig2 = plot_episodes(policy_image, parameter, 'Policy')
fig2.savefig(f'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\TD_learning\\alpha\\alpha_1\\Episodes_policy_{parameter}.png')

utils.plot_value_function(final_values, maze)
utils.plot_policy(final_values, maze)

# animation_gif('C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots', 9998, 9999)

class MazeQLearning(MazeEnvironment): #Inherited from MazeEnvironment
    def __init__(self, maze, alpha = 0.1, gamma = 0.95, epsilon = 0.2, episodes = 10000):
        super().__init__()
        
        self.maze = maze
        self.alpha = alpha #Learning Rate
        self.gamma = gamma #Discount factor
        self.epsilon = epsilon #Exploration Rate
        self.episodes = episodes
        self.q_table = np.zeros((*maze.shape, 4))  # Assuming 4 actions: up, down, left, right
        
        self.convergence_history = []
    def choose_action(self,state):
        
        #Exploration:
        if np.random.rand() < self.epsilon: #any action
            return np.random.choice(list(self.actions.keys()))
        
        #Exploitation:
        else:
            #Geting the utilities of all possible actions from current state
            return np.argmax(self.q_table[state[0], state[1]])
    
    def update_q_table(self, action, current_state, reward, new_state):
        current_q = self.q_table[current_state[0],current_state[1], action]
        max_future_q = max(self.q_table[new_state[0], new_state[1]])
        new_q = current_q + self.alpha * (reward + self.gamma * (max_future_q) - current_q)
        self.q_table[current_state[0], current_state[1], action] = new_q
        
    def run_episodes(self):
        
        selected_episodes = [0, 49, 99, 999, 4999, 9999]
        utility_filenames = []
        policy_filenames = []
        #mask = (maze != 2) & (maze != 3)
        
        for episode in range(self.episodes):
            
            print("Episode: ", episode)
            # starting_episode = 9998
            # ending_episode = 9999
            # i = 0
            prev_utility = np.copy(np.max(self.q_table, axis=2)) #before the update
            
            current_state = self.reset()
            done = False
            while not done:
                
                #Animation purposes
                # if (starting_episode <= episode <= ending_episode):
                #     i += 1
                #     plt.clf()
                #     plt.imshow(self.maze, cmap='hot', interpolation='nearest')
                #     plt.scatter(self.current_pos[1], self.current_pos[0], c='blue', s=100)  # Agent's position
                #     plt.title(f'QL_Maze_{episode}')
                #     if i<=9:
                #         plt.savefig(f"C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots\\figure_000{i}")   # save the figure to file
                #     elif 10 <= i <= 99:
                #         plt.savefig(f"C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots\\figure_00{i}")   # save the figure to file
                #     elif 100 <= i <= 999:
                #         plt.savefig(f"C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots\\figure_0{i}")   # save the figure to file
                #     else:
                #         plt.savefig(f"C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots\\figure_{i}")   # save the figure to file
                
                
                action = self.choose_action(current_state)
                new_state, reward, done = self.step(action)
                self.update_q_table(action, current_state, reward, new_state)
                current_state = new_state
                
                # #This part is for initialization of the goal and trap utility
                # if (self.maze[new_state] == 3) and (max(self.q_table[new_state[0], new_state[1]]) == 0): 
                #     self.q_table[new_state[0], new_state[1]] = 1000 #goal
                # if (self.maze[new_state] == 2) and (max(self.q_table[new_state[0], new_state[1]]) == 0): 
                #     self.q_table[new_state[0], new_state[1]] = -1000 #trap
                                    
            diff = np.sum(np.abs(np.copy(np.max(self.q_table, axis=2)) - prev_utility))
            self.convergence_history.append(diff)
                    
            # if episode in selected_episodes:
                    
            #     ax = utils_update.plot_value_function(copy.deepcopy(np.max(self.q_table, axis=2)), np.copy(self.maze), episode)
            #     filename_1 = f'Utility_Episode_{episode + 1}_alpha_0_001.png'
            #     ax.figure.savefig(filename_1)
            #     utility_filenames.append(filename_1)
            #     plt.close()
                
            #     ax = utils_update.plot_policy(copy.deepcopy(np.max(self.q_table, axis=2)), np.copy(self.maze), episode)
            #     filename_2 = f'Policy_Episode_{episode + 1}_alpha_0_001.png'
            #     ax.figure.savefig(filename_2)
            #     policy_filenames.append(filename_2)
            #     plt.close()
                
        return self.q_table, utility_filenames, policy_filenames

alpha = 0.1
gamma = 0.95
epsilon = 1
      
maze_q_learning = MazeQLearning(maze, alpha, gamma, epsilon, episodes=10000)
final_q_table, utility_image, policy_image = maze_q_learning.run_episodes()

final_utilities = np.max(final_q_table, axis=2) # Take max across the action dimension

history = maze_q_learning.convergence_history
#history_smoothed = moving_average(history, 50)
history_smoothed = exponential_moving_average(history, 0.02)

parameter = 'epsilon = 1'
plt.figure(figsize=(20, 10))
episodes = np.arange(1,10001) # Episodes from 1 to 10000
plt.plot(episodes,history_smoothed)
plt.xlabel('Episode')
plt.ylabel('Sum of Absolute Differences')
plt.title(f'Convergence of Value Function for {parameter}')
plt.grid(True)
plt.savefig(f'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\Q_learning\\epsilon\\epsilon_1\\Convergence_for_parameter_{parameter}.png')
plt.show()

fig1 = plot_episodes(utility_image, parameter, 'Utility')
fig1.savefig(f'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\Q_learning\\epsilon\\epsilon_1\\Episodes_utility_{parameter}.png')

fig2 = plot_episodes(policy_image, parameter, 'Policy')
fig2.savefig(f'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\Q_learning\\epsilon\\epsilon_1\\Episodes_policy_{parameter}.png')

final_utilities[maze == 1] = -1000 #boundaries
# animation_gif('C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW3\\animation_plots', 9998, 9999)
utils.plot_value_function(final_utilities, maze)
utils.plot_policy(final_utilities, maze)
# Not finished

# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================================================================
#                                     Main
# =============================================================================


def calculate_angle(current_position, last_position):
    # Calculate the displacement vector between current and last positions
    displacement = current_position - last_position

    # Calculate the angle in radians using arctan2 function
    angle_radians = np.arctan2(displacement[1], displacement[0])

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


# Create the custom environment
class CleaningRobotEnv(gym.Env):
    def __init__(self):
        # Define the state space and action space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,))
        self.action_space = gym.spaces.Discrete(4)  # Four possible actions: up, down, left, right

        # Define the initial position of the robot
        self.robot_position = np.array([0.5, 0.5])
        self.robot_velocity = np.array([0.02, 0.02])

        # Define the positions of the small dots (mass) (x,y), (x,y)
        self.mass_positions = np.array([[0.4, 0.4], [0.8, 0.7], [0.6, 0.1]])

    def print_title(self, angle, reward):
        print()
        print(f"Robot Position: {self.robot_position}")
        print(f"Robot Velocity: {self.robot_velocity}")
        print(f"Angle: {round(angle, 2)}º")
        print(f"Reward: {round(reward, 2)}º")

    def reset(self):
        # Reset the robot position and velocity
        self.robot_position = np.array([0.5, 0.5])
        self.robot_velocity = np.array([0.02, 0.02])
        return self.robot_position

    def step(self, action):

        last_robot_position = self.robot_position.copy()

        # Update the robot's velocity based on the chosen action
        if action == 0:  # Up
            self.robot_velocity[1] += 0.01
        elif action == 1:  # Down
            self.robot_velocity[1] -= 0.01
        elif action == 2:  # Left
            self.robot_velocity[0] += 0.01
        elif action == 3:  # Right
            self.robot_velocity[0] -= 0.01
        elif action == 4:  # Continue
            self.robot_velocity[0] += 0.0

        # Update the robot's position based on the velocity
        self.robot_position += self.robot_velocity

        # Bounce the robot when it touches the walls
        if self.robot_position[0] < 0 or self.robot_position[0] > 1:
            self.robot_velocity[0] *= -1
        if self.robot_position[1] < 0 or self.robot_position[1] > 1:
            self.robot_velocity[1] *= -1

        angle = calculate_angle(self.robot_position, last_robot_position)

        # Calculate the reward based on the distance to the closest small dot (mass)
        closest_mass = np.min(np.linalg.norm(self.robot_position - self.mass_positions, axis=1))
        reward = - 5 * closest_mass  # Negative reward based on proximity to the closest mass

        # Check if the distance between the robot and the small dot (mass) is less than 1
        if closest_mass < 0.1:
            reward += 10  # Additional positive reward
            done = True
        else:
            done = False

        # Check if the robot has reached a small dot (mass)
        if np.any(np.all(np.isclose(self.robot_position, self.mass_positions), axis=1)):
            reward += 10  # Positive reward for reaching a mass

        self.print_title(angle, reward)

        return self.robot_position, reward, done, {}


# Define the Deep Q-Network (DQN) agent
class DQNAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = 1.0  # Exploration rate
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.001  # Learning rate

        # Initialize the DQN model
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.state_space,), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Choose a random action for exploration
            return np.random.choice(self.action_space)
        else:
            # Use the DQN model to choose the best action based on current state
            return np.argmax(self.model.predict(state))

    def train(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.amax(self.model.predict(next_state))
        target_full = self.model.predict(state)
        target_full[0][action] = target
        self.model.fit(state, target_full, epochs=1, verbose=0)

        if done:
            # Update exploration rate (epsilon) over time
            if self.epsilon > 0.01:
                self.epsilon *= 0.995


# Create the environment and agent
env = CleaningRobotEnv()
agent = DQNAgent(state_space=2, action_space=4)

# Create the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
robot_patch = ax.add_patch(Circle((0, 0), 0.05, color='r'))  # Plot robot position
mass_scatter = ax.scatter([], [], color='b')  # Plot mass positions
plt.title(f"Cleaning Robot\nReward: {0}")
plt.xlabel("X")
plt.ylabel("Y")
plt.draw()
plt.show(block=False)

# Train the agent
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, 2])
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 2])
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Update the plot with the robot's position and mass positions
        robot_patch.center = (state[0][0], state[0][1])
        mass_scatter.set_offsets(env.mass_positions)
        plt.title(f"Cleaning Robot\nReward: {round(reward,2)}")
        plt.pause(0.1)

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Test the trained agent
state = env.reset()
state = np.reshape(state, [1, 2])
done = False
total_reward = 0

while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, 2])
    state = next_state
    total_reward += reward

print(f"Test Total Reward: {total_reward}")

"""
We import the necessary libraries,
including gym for creating the custom environment, numpy for numerical operations,
and keras for building the Deep Q-Network (DQN) agent.

We define a custom environment called CleaningRobotEnv that inherits from the gym.Env class.
This environment represents the room where the cleaning robot operates.
It has methods for resetting the environment (reset) and performing actions (step).
The step method moves the robot based on the chosen action, calculates the reward based on
proximity to the nearest mass, and checks if the robot has reached a mass.

We define the DQN agent in the DQNAgent class.
The agent initializes the DQN model with two hidden layers and an output layer.
It also has methods for choosing actions (act) based on exploration-exploitation and
training the model (train) using the DQN algorithm.

We create an instance of the CleaningRobotEnv environment and an instance of the DQNAgent agent.

We train the agent by running multiple episodes. In each episode, we reset the environment,
perform actions, and update the agent's Q-values using the DQN algorithm.
We also gradually decrease the exploration rate (epsilon) over time to shift from exploration to exploitation.

After training, we test the trained agent by running a test episode. We reset the environment
and let the agent choose actions based on the learned policy.
We calculate the total reward obtained during the test episode.

By following this code, the DQN agent learns to navigate the continuous space,
explore the room, and find the small dots (mass).
The agent is trained through interaction with the environment,
adjusting its actions based on the observed rewards, and updating the Q-values to optimize its policy.
"""

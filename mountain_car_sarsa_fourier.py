import gym, itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

# Initializations
num_episodes = 3000  # 1000
num_timesteps = 600  # 200
gym.envs.register(
    id="MountainCarLongerEpisodeLength-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=num_timesteps,  # MountainCar-v0 uses 200
    reward_threshold=-110.0,
)
env = gym.make("MountainCarLongerEpisodeLength-v0")
num_actions = env.action_space.n
dim = env.observation_space.high.size

# Parameters
# order of the basis functions
order = 3
# total number of such basis
num_ind = int(pow(order + 1.0, dim))
# multipliers are the coefficient vectors used within the computation of cosine or sine function computation
multipliers = np.zeros((num_ind, dim))

epsilon = 0.1
Lambda = 0.5
alpha = 0.00005
gamma = 0.99


xbar = np.zeros((2, dim))
xbar[0, :] = env.observation_space.low
xbar[1, :] = env.observation_space.high

ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)

# These are the weights which the basis functions are multiplied to get a value
weights = np.zeros((num_ind, num_actions))

# this function returns a normalized state where all variable values are between 0 and 1.
def normalize_state(s):
    y = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] > xbar[1, i]:
            y[i] = 1
        elif s[i] < xbar[0, i]:
            y[i] = 0
        else:
            y[i] = (s[i] - xbar[0, i]) / (xbar[1, i] - xbar[0, i])
    return y


# Returns an 1D array of basis functions
def phi(state):
    # Initialize an empty array for the basis functions
    basis_functions = []

    # Iterate through each dimension
    for i in range(len(state)):
        for j in range(order + 1):
            # Compute the Fourier basis functions using sine and cosine functions
            basis_functions.append(math.cos(j * math.pi * state[i]))
            basis_functions.append(math.sin(j * math.pi * state[i]))

    # Convert the basis functions into a NumPy array
    return np.array(basis_functions)


# Create the fourier basis coefficients
def create_multipliers():
    global multipliers

    # Generate all possible combinations of multipliers for each dimension
    combinations = list(itertools.product(range(order + 1), repeat=dim))

    # Initialize the multipliers array
    multipliers = np.zeros((num_ind, dim))

    # Assign the combinations to the multipliers
    for i, combo in enumerate(combinations):
        multipliers[i, :] = combo


# Returns the value of an action at some state
def action_value(input_state, action):
    state_normalized = normalize_state(input_state)
    features = phi(state_normalized)
    Qvalue = np.dot(weights[:, action], features)
    return Qvalue


# Returns an exploratory action for an input
def learning_policy(input_state):
    global weights
    state_normalized = normalize_state(input_state)
    features = phi(state_normalized)
    Qvalues = np.dot(features, weights)
    random_num = np.random.random()
    if random_num < 1. - epsilon:
        action_chosen = Qvalues.argmax()
    else:
        action_chosen = env.action_space.sample()
    return int(action_chosen)


def test_policy(state):
    global weights
    state_normalized = normalize_state(state)
    features = phi(state_normalized)
    Qvalues = np.dot(features, weights)
    return int(Qvalues.argmax())


def SARSA_Learning():
    """
        Implement Sarsa-lambda algorithm to update qtable, etable and learning policy.
        Input:
            all parameters
            
        Output: 
            This function does not return an output. It only updates the weights according to the algorithm.
    
    """
    global weights
    for ep in range(num_episodes):
        print("Episode", ep+1, "/", num_episodes)
        e = np.zeros((num_ind, num_actions))
        state = env.reset()[0]
        norm_state = normalize_state(state)
        features = phi(norm_state)
        action = learning_policy(state)
        rewards = np.array([0])

        # Each episode
        for t in range(num_timesteps):
            env.render()
            next_state, reward, done, _, info = env.step(action)
            rewards = np.append(rewards, reward)

            # Normalize the next state and compute its features
            norm_next_state = normalize_state(next_state)
            next_features = phi(norm_next_state)

            # Compute the TD error
            td_error = reward + gamma * np.dot(weights[:, action], next_features) - np.dot(weights[:, action], features)

            # Update the eligibility traces
            e *= gamma * Lambda
            e[:, action] += features

            # Update the weights
            weights[:, action] += alpha * td_error * e[:, action]

            # Update the state and action
            state = next_state
            norm_state = norm_next_state
            features = next_features

            # Choose the next action using the learning policy
            action = learning_policy(state)

            if done:
                break

        ep_length[ep] = t
        print(np.sum(rewards))
    np.save("mountain_car_saved_weights_grading.npy", weights)

def SARSA_Test(num_test_episodes):
    rewards_each_test_episode = []
    steps_each_test_episode = []
    for _ in range(num_test_episodes):
        state = env.reset()[0]
        rewards = np.array([0])
        steps = 0
        # Each episode
        for _ in range(num_timesteps):
            env.render()
            action = test_policy(state)
            next_state, reward, done, _, _ = env.step(action)
            steps += 1
            state = next_state
            rewards = np.append(rewards, reward)
            if done:
                # print((rewards.shape))
                break
        rewards_each_test_episode.append(np.sum(rewards))
        steps_each_test_episode.append(steps)
    print("Rewards each test episode: {}".format(rewards_each_test_episode))
    print("Steps each test episode: {}".format(steps_each_test_episode))
    # print(np.sum(rewards))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        "*** run without input arguments to learn ***"
        create_multipliers()
        SARSA_Learning()
    elif sys.argv[1] == "test":
        "*** run to test the saved policy with input argument test ***"
        create_multipliers()
        weights = np.load("mountain_car_saved_weights_grading.npy")
        num_test_episodes = 100
        SARSA_Test(num_test_episodes)
    else:

        print("unknown input argument")

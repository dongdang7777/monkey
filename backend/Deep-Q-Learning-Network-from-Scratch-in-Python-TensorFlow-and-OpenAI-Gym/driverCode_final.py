"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

This file contains driver code that imports DeepQLearning class developed in the file "functions_final"
 
The class DeepQLearning implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
The implementation is based on the OpenAI Gym Cart Pole environment and TensorFlow (Keras) machine learning library

The webpage explaining the codes and the main idea of the DQN is given here:

https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/


Author: Aleksandar Haber 
Date: February 2023

Tested on:

tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.11.0
tensorflow-estimator==2.11.0
tensorflow-intel==2.11.0
tensorflow-io-gcs-filesystem==0.30.0

keras==2.11.0

gym==0.26.2

"""
# import the class
import numpy as np
import tensorflow as tf
from env import fighterEnv
from functions_final import DeepQLearning
# classical gym 
import gym
# instead of gym, import gymnasium 
#import gymnasium as gym

# create environment
env=fighterEnv();
input_shape=20
num_actions=11

policy_network = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='softmax')
])


# Set up the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
# select the parameters

# Set up lists to store episode rewards and lengths
episode_rewards = []
episode_lengths = []

num_episodes = 1000
discount_factor = 0.99

# Train the agent using the REINFORCE algorithm
for episode in range(num_episodes):
    # Reset the environment and get the initial state
    state = env.reset()
    episode_reward = 0
    episode_length = 0

    # Keep track of the states, actions, and rewards for each step in the episode
    states = []
    actions = []
    rewards = []

    # Run the episode
    while True:
        # Get the action probabilities from the policy network
        print(np.array())
        action_probs = policy_network.predict(np.array([state]))[0]
        
        # Choose an action based on the action probabilities
        action = np.random.choice(num_actions, p=action_probs)

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # Store the current state, action, and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        # Update the current state and episode reward
        state = next_state
        episode_reward += reward
        episode_length += 1

        # End the episode if the environment is done
        if done:
            print('Episode {} done !!!!!!'.format(episode))
            break

    # Calculate the discounted rewards for each step in the episode
    discounted_rewards = np.zeros_like(rewards)
    running_total = 0
    for i in reversed(range(len(rewards))):
        running_total = running_total * discount_factor + rewards[i]
        discounted_rewards[i] = running_total

    # Normalize the discounted rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    # Convert the lists of states, actions, and discounted rewards to tensors
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards)

    # Train the policy network using the REINFORCE algorithm
    with tf.GradientTape() as tape:
        # Get the action probabilities from the policy network
        action_probs = policy_network(states)
        # Calculate the loss
        loss = tf.cast(tf.math.log(tf.gather(action_probs,actions,axis=1,batch_dims=1)),tf.float64)
        
        loss = loss * discounted_rewards
        loss = -tf.reduce_sum(loss)

    # Calculate the gradients and update the policy network
    grads = tape.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

    # Store the episode reward and length
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    policy_network.save('trained_model_temp.keras')
# number of training episodes
# NOTE HERE THAT AFTER CERTAIN NUMBERS OF EPISODES, WHEN THE PARAMTERS ARE LEARNED
# THE EPISODE WILL BE LONG, AT THAT POINT YOU CAN STOP THE TRAINING PROCESS BY PRESSING CTRL+C
policy_network.summary()
# save the model, this is important, since it takes long time to train the model 
# and we will need model in another file to visualize the trained model performance
policy_network.save("trained_model_temp.keras")




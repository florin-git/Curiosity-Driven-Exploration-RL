import numpy as np
import random

import torch
import threading

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from agent import *
from envs import *
from utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# For reproducibility
SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


use_curiosity = True
num_workers = 8  
global_epochs = 20
training_epochs = 50
n_step = 128  # step rollout
# n_step = 256
SAVE_EVERY = 5
gamma = 0.99

batch_size = 32
hidden_size = 256
learning_rate = 2e-4


envs = [create_environment() for _ in range(num_workers)]
data = [None for _ in range(num_workers)]


def thread_routine(idx, action):
    global envs
    env = envs[idx]
    state, reward, done, _, info = env.step(action)

    if done:
        state, _ = env.reset()
    global data
    data[idx] = [state, reward, done]


def send_actions(n, actions):
    threads = []
    for i in range(n):
        t = threading.Thread(target = thread_routine, args = (i, actions[i]))
        t.start()
        threads.append((i,t))
    return threads


def get_mean_std_from_states(n_actions, normalization_steps=150):
    next_states = []
    steps = 0
    
    global envs
    for environ in envs:
        environ.reset()

    while steps < normalization_steps:
        steps += num_workers

        # Randomly choose an action for each worker
        actions = np.random.randint(0, n_actions, size=(num_workers,))

        workers = send_actions(num_workers, actions)

        global data
        for idx, w in workers:
            w.join()
            ns, r, d = data[idx]
            next_states.append(ns)

    next_states = np.stack(next_states)
    states_mean = np.mean(next_states, axis=0)
    states_std = np.std(next_states, axis=0)

    return states_mean, states_std


def standardize_array(array, mean, std):
    return (array - mean) / (std + 1e-8)


def main():
    env = create_environment()
    n_actions = env.action_space.n

    agent = Agent(n_actions=n_actions, device=device,
                  use_icm=use_curiosity, batch_size=batch_size, 
                  hidden_size=hidden_size, learning_rate=learning_rate,
                  epochs=training_epochs)
    

    ext_reward_deque = deque(maxlen=3)
    int_reward_deque = deque(maxlen=3)
    loss_deque = deque(maxlen=3)

    global envs
    for environ in envs:
        environ.reset()

    # Initial state for every worker
    state, _ = env.reset()

    states = [state for _ in range(num_workers)]

    for e in range(global_epochs):
        # histories to make training data
        h_states, h_extrinsic_r, h_intrinsic_r, h_dones, h_next_states, h_actions, h_values, h_policies = [
        ], [], [], [], [], [], [], []

        global data
        data = [None for _ in range(num_workers)]

        # rollout (one for every worker)
        for i in range(n_step):
 
            actions, value, policy = agent.sample_action(states)

            workers = send_actions(num_workers, actions)

            next_states = [None for _ in range(num_workers)]
            rewards = [None for _ in range(num_workers)] 
            dones = [None for _ in range(num_workers)] 

            for idx, w in workers:
                w.join()
                ns, r, d = data[idx]
                next_states[idx] = ns
                rewards[idx] = r
                dones[idx] = d

            rewards = np.array(rewards)
            dones = np.array(dones)

            intrinsic_reward = agent.intrinsic_reward(
                states, 
                next_states,
                actions
            )

            h_intrinsic_r.append(intrinsic_reward)
            h_states.append(states)
            h_next_states.append(next_states)
            h_extrinsic_r.append(rewards)
            h_dones.append(dones)
            h_actions.append(actions)
            h_values.append(value)
            h_policies.append(policy)

            states = next_states.copy()

        # Compute last next value
        _, value, _ = agent.sample_action(states)


        h_values.append(value)

        # running mean intrinsic reward

        # total_state: [num_workers * n_step, state.shape], i.e. [num_workers * n_step, 4, 84, 84]
        total_state = np.stack(h_states).transpose(
            [1, 0, 2, 3, 4]).reshape(-1, *state.shape)
        total_next_state = np.stack(h_next_states).transpose(
            [1, 0, 2, 3, 4]).reshape(-1, *state.shape)

        # total_intrinsic_r: [num_workers, n_step]
        total_intrinsic_r = np.stack(h_intrinsic_r).T  # It is transposed
        total_extrinsic_r = np.stack(h_extrinsic_r).T
        total_extrinsic_r *= 1e-2

        total_action = np.stack(h_actions).T.reshape(-1)
        total_value = np.stack(h_values).T
        total_done = np.stack(h_dones).T
        # total_policy = np.stack(h_policies).reshape(-1, n_actions)


        # Make target and advantage
        target, advantage = compute_target_advantage(
            total_intrinsic_r + total_extrinsic_r,
            # total_intrinsic_r,
            total_done,
            total_value,
            gamma,
            n_step,
            num_workers
        )

        advantage = standardize_array(
            advantage, np.mean(advantage), np.std(advantage))
        
        target = standardize_array(
            target, np.mean(target), np.std(target))

        loss = agent.train(total_state,
                           total_next_state,
                           target,
                           total_action,
                           advantage,
                           h_policies)

        loss_deque.append(loss)
        ext_reward_deque.append(np.mean(total_extrinsic_r))
        int_reward_deque.append(np.mean(total_intrinsic_r))

        # save model
        if e % SAVE_EVERY == 0 or e == global_epochs - 1:
            if agent.use_icm == True:  # modifica per fare fine tuning se necessario
                torch.save(agent.model.state_dict(),
                           './models/agent_with_curiosity.pt')
            else:
                torch.save(agent.model.state_dict(), './models/agent.pt')

        print(f"\nepoch {e + 1}: loss = {np.mean(loss_deque):.3e}, reward = {np.mean(ext_reward_deque):.3f}, curiosity = {np.mean(int_reward_deque):.3e}\n")



if __name__ == '__main__':
    main()

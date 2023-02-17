import numpy as np

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from models import *


class Agent:
    def __init__(
        self,
        n_actions,
        device,
        use_icm=True,
        epochs=10,
        batch_size=16,
        hidden_size=512,
        learning_rate=2e-4,
        epsilon_ppo=0.3,
        beta=0.2,
        eta=2,
    ):

        self.device = device
        self.use_icm = use_icm
        self.n_actions = n_actions
        self.epochs = epochs

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.epsilon_ppo = epsilon_ppo
        self.beta = beta
        self.eta = eta

        # Actor-Critic
        self.model = ActorCritic(
            hidden_size, self.n_actions, self.device).to(self.device)
        self.model_optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate)

        # Curiosity
        if use_icm:
            self.icm = ICM(hidden_size, self.n_actions,
                           self.model.state_features_extractor, self.device).to(self.device)
            self.icm_optimizer = optim.Adam(
                self.icm.parameters(), lr=learning_rate)

    def intrinsic_reward(self, state, next_state, action):
        if not self.use_icm:
            return np.zeros_like(action)
        
        next_state_feats, pred_next_state_feats, _, _ = self.icm(
            state, next_state, action)
        
        # shape: [num_workers] Necessary if we have multiple workers
        intrinsic_reward = self.eta * \
            F.mse_loss(pred_next_state_feats, next_state_feats,
                       reduction='none').mean(-1)
        return intrinsic_reward.data.cpu().numpy()

    def sample_action(self, state):  # returns an action sampled from the policy in that state
        policy, value = self.model(state)
        m = Categorical(F.softmax(policy, dim=-1))
        action = m.sample()
        
        return action.data.cpu().numpy(), value.data.cpu().numpy().squeeze(), policy.detach()

    def train(self, states, next_states, targets, actions, advs, old_policies):

        if self.use_icm:
            cross_entropy = nn.CrossEntropyLoss()
            forward_model_MSE = nn.MSELoss()

        advs = torch.FloatTensor(advs).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)

        with torch.no_grad():
            policy_old_list = torch.stack(old_policies).permute(
                1, 0, 2).contiguous().view(-1, self.n_actions).to(self.device)
            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(torch.tensor(
                actions, dtype=torch.float32, device=self.device))

        sample_range = np.arange(len(states))
        for i in range(self.epochs):
            np.random.shuffle(sample_range)
            for j in range(int(len(states) / self.batch_size)):
                idx = sample_range[j*self.batch_size: (j + 1)*self.batch_size]

                inverse_dynamics_loss, forward_loss = 0, 0

                if self.use_icm:
                    nstate, pred_nstate, action, pred_action = self.icm(
                        states[idx], next_states[idx], actions[idx])
                    
                    inverse_dynamics_loss = cross_entropy(pred_action, action)
                    forward_loss = forward_model_MSE(pred_nstate, nstate)

                policy, value = self.model(states[idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(torch.tensor(
                    actions[idx], dtype=torch.float32, device=self.device))

                ratio = torch.exp(log_prob - log_prob_old[idx])

                surr1 = ratio * advs[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_ppo,
                                    1.0 + self.epsilon_ppo) * advs[idx]

                actor_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    value = value.squeeze()


                critic_loss = F.mse_loss(value, targets[idx])
                entropy = m.entropy().mean()

                self.model_optimizer.zero_grad()
                if self.use_icm:
                    self.icm_optimizer.zero_grad()

                loss = 0.1 * (actor_loss + 0.5 * critic_loss - 0.001 * entropy) + \
                    (1 - self.beta)*inverse_dynamics_loss + self.beta* forward_loss
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.model_optimizer.step()

                if self.use_icm:
                    torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 0.5)
                    self.icm_optimizer.step()

            if i % 10 == 0 or i == self.epochs - 1:
                print(
                    f"\ttraining step {i + 1}: actor {actor_loss:.3f} critic {critic_loss:.3e} entropy {entropy:.3f} inverse {inverse_dynamics_loss:.3f} forward {forward_loss:.3e}")

            

        return loss.item()

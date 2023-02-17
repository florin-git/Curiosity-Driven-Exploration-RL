import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Flatten(nn.Module):
  def forward(self, state):
    return state.reshape(-1, 16*21*21)

# class ICM(nn.Module): #intrinsic curiosity module
#   def __init__(self, hidden_size, n_actions, device):
#     super(ICM, self).__init__()
#     self.hidden_size = hidden_size
#     self.n_actions = n_actions
#     self.device = device

#     self.state_features_extractor = nn.Sequential(#può essere la stessa rete di Actor critic??? (parameter sharing)
#         nn.Conv2d(4, 16, 7, padding = 3),
#         nn.ReLU(),
#         nn.Conv2d(16, 16, 7, padding = 3),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(16, 16, 5, padding = 2),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         Flatten(),
#         nn.Linear(16*21*21, hidden_size),
#         nn.ReLU()
#     )
    
#     self.inverse_dynamics_model = nn.Sequential(
#         nn.Linear(2*hidden_size, hidden_size),
#         nn.ReLU(),
#         nn.Linear(hidden_size, n_actions),
#     )

#     # forse meglio avere una residual connection
#     self.forward_model = nn.Sequential(
#         nn.Linear(hidden_size + n_actions, hidden_size),
#         nn.ReLU(),
#         nn.Linear(hidden_size, hidden_size),
#         nn.ReLU()
#     )

#   def input_preprocessing(self, state, next_state, action):#valuta se lasciarla qui o metterla in Agent
#     state = torch.FloatTensor(np.array([s.__array__() for s in state])).to(self.device)
#     next_state = torch.FloatTensor(np.array([s.__array__() for s in next_state])).to(self.device)
#     action = torch.tensor(action).to(self.device)
#     action = F.one_hot(action, num_classes = self.n_actions).type('torch.FloatTensor').view(-1, self.n_actions).to(self.device)
    
#     return state, next_state, action

#   def forward(self, state, next_state, action):
#     state, next_state, action = self.input_preprocessing(state, next_state, action)
#     state = self.state_features_extractor(state)
#     next_state = self.state_features_extractor(next_state)

#     predicted_action = self.inverse_dynamics_model(torch.cat((state, next_state), dim=1)) #logits pre softmax 
#     predicted_next_state = self.forward_model(torch.cat((state, action), dim=1))# self.forward_model(state, action) se torniamo ForwardModel

#     return next_state, predicted_next_state, action, predicted_action

# class ActorCritic(nn.Module):
#   def __init__(self, hidden_size, n_actions, device):
#     super(ActorCritic, self).__init__()
#     self.hidden_size = hidden_size
#     self.n_actions = n_actions
#     self.device = device
  
#     self.state_features_extractor = nn.Sequential(#può essere la stessa rete di ICM??? (parameter sharing)
#         nn.Conv2d(4, 16, 7, padding = 3),
#         nn.ReLU(),
#         nn.Conv2d(16, 16, 7, padding = 3),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(16, 16, 5, padding = 2),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         Flatten(),
#         nn.Linear(16*21*21, hidden_size),
#         nn.ReLU()
#     )

#     self.actor = nn.Sequential(
#         nn.Linear(hidden_size, hidden_size),
#         nn.ReLU(),
#         nn.Linear(hidden_size, n_actions),
#     )

#     self.critic = nn.Sequential(
#         nn.Linear(hidden_size, hidden_size),
#         nn.ReLU(),
#         nn.Linear(hidden_size, 1),
#     )
  
#   def forward(self, state):
#     state = torch.FloatTensor(np.array([s.__array__() for s in state])).to(self.device)
#     state = self.state_features_extractor(state)
#     policy = self.actor(state) # (logits pre softmax)
#     value = self.critic(state) # V(state)
#     return policy, value
  
#   def act(self, states):
#     states = torch.vstack([s for s in states]).to(self.device)
#     action_prob = F.softmax(self.actor(self.state_features_extractor(states)), dim=-1)
#     return action_prob.argmax(-1).cpu().item()

class FeatureExtractor(nn.Module):
  def __init__(self, hidden_size):
    super(FeatureExtractor, self).__init__()
    
    self.cnn = nn.Sequential(
        nn.Conv2d(4, 16, 7, padding = 3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, 7, padding = 3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 5, padding = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16*21*21, hidden_size),
    )

    # self.cnn = nn.Sequential(
    #     nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
    #     nn.BatchNorm2d(32),
    #     nn.ReLU(),
    #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
    #     nn.BatchNorm2d(64),
    #     nn.ReLU(),
    #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
    #     nn.BatchNorm2d(64),
    #     nn.ReLU(),
    #     nn.Flatten(),
    # )

    # out_features_after_cnn = np.prod(self.cnn(torch.rand(1, *(4,84,84))).shape)
    
    # self.linear = nn.Sequential(
    #     # nn.Linear(out_features_after_cnn, 1024),
    #     # nn.ReLU(),
    #     # nn.Linear(1024, hidden_size)
    #     nn.Linear(out_features_after_cnn, hidden_size),
    # )
    
    for p in self.modules():
      if isinstance(p, nn.Conv2d):
        init.orthogonal_(p.weight, np.sqrt(2))
        # init.kaiming_normal_(p.weight, mode='fan_out', nonlinearity='relu')
        if p.bias is not None:
            init.constant_(p.bias, 0)

      if isinstance(p, nn.Linear):
        init.orthogonal_(p.weight, np.sqrt(2))
        # init.xavier_normal_(p.weight)
        if p.bias is not None:
            nn.init.constant_(p.bias, 0)
  
  def forward(self, states):
    # states = self.cnn(states)
    # return self.linear(states)

    return self.cnn(states)
  

class ICM(nn.Module): #intrinsic curiosity module
  def __init__(self, hidden_size, n_actions, cnn, device):
    super(ICM, self).__init__()
    self.hidden_size = hidden_size
    self.n_actions = n_actions
    self.device = device

    self.state_features_extractor = cnn
    
    self.inverse_dynamics_model = nn.Sequential(
        nn.Linear(2*hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions),
    )

    self.forward_model = nn.Sequential(
        # nn.Linear(hidden_size + n_actions, hidden_size),
        nn.Linear(hidden_size + 1, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size)
    )

  def input_preprocessing(self, state, next_state, action):#valuta se lasciarla qui o metterla in Agent
    state = torch.FloatTensor(np.array([s.__array__() for s in state])).to(self.device)
    next_state = torch.FloatTensor(np.array([s.__array__() for s in next_state])).to(self.device)
    action = torch.tensor(action, dtype=torch.int64).to(self.device)
    # action = F.one_hot(action, num_classes = self.n_actions).type('torch.FloatTensor').view(-1, self.n_actions).to(self.device)
    return state, next_state, action

  def forward(self, state, next_state, action):
    state, next_state, action = self.input_preprocessing(state, next_state, action)
    
    state = self.state_features_extractor(state)
    next_state = self.state_features_extractor(next_state)

    predicted_action = self.inverse_dynamics_model(torch.cat((state, next_state), 1)) #logits pre softmax 
    # predicted_next_state = self.forward_model(torch.cat((state, action), 1))
    predicted_next_state = self.forward_model(torch.cat((state, action.unsqueeze(1)), 1))

    return next_state, predicted_next_state, action, predicted_action




class ActorCritic(nn.Module):
  def __init__(self, hidden_size, n_actions, device):
    super(ActorCritic, self).__init__()
    self.hidden_size = hidden_size
    self.n_actions = n_actions
    self.device = device
  
    self.state_features_extractor = FeatureExtractor(hidden_size).to(self.device)

    self.actor = nn.Sequential(
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions),
    )

    self.critic = nn.Sequential(
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        # nn.Linear(hidden_size, 1),
        nn.Linear(hidden_size, hidden_size//2),
        nn.ReLU(),
        nn.Linear(hidden_size//2, 1)
    )

 
    for i in range(len(self.actor)):
      if type(self.actor[i]) == nn.Linear:
        # init.uniform_(self.actor[i].weight, -initrange, initrange)
        init.orthogonal_(self.actor[i].weight, 0.01)
        self.actor[i].bias.data.zero_()

    
    for i in range(len(self.critic)):
      if type(self.critic[i]) == nn.Linear:
        # init.uniform_(self.critic[i].weight, -initrange, initrange)
        init.orthogonal_(self.critic[i].weight, 0.01)
        self.critic[i].bias.data.zero_()

    # for p in self.actor.modules():
    #   if isinstance(p, nn.Linear):
    #     init.xavier_normal_(p.weight)
    #     if p.bias is not None:
    #         nn.init.constant_(p.bias, 0)

    # for p in self.critic.modules():
    #   if isinstance(p, nn.Linear):
    #     init.xavier_normal_(p.weight)
    #     if p.bias is not None:
    #         nn.init.constant_(p.bias, 0)
  
  def forward(self, state):
    state = torch.FloatTensor(np.array([s.__array__() for s in state])).to(self.device)
    state = self.state_features_extractor(state)

    policy = self.actor(state) # (logits pre softmax)
    value = self.critic(state) # V(state)
    return policy, value

  def act(self, states): #action that maximizes the reward (for evaluation)
    # states = torch.vstack([s for s in states]).to(self.device)
    # 'unsqueeze' is used to have a batch of size 1
    states = torch.vstack([s for s in states]).unsqueeze(0).to(self.device)
    action_logits = self.actor(self.state_features_extractor(states))
    return action_logits.argmax(-1).cpu().item()
  
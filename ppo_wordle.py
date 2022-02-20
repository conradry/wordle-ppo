import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import scipy.signal
from wordle import WordleEnv

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(self, obs_dim, size, gamma=0.99, lamb=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lamb
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

class MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU(inplace=True)

    def _distribution(self, state):
        logits = self.run_net(state)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, dist, action):
        return dist.log_prob(action)

    def run_net(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim, emb_dim):
        super().__init__()
        self.pi = MLP(obs_dim, hidden_dim, act_dim)
        self.v  = MLP(obs_dim, hidden_dim, 1)
        self.w_embed = nn.Embedding(act_dim, emb_dim)
        self.s_embed = nn.Embedding(4, 3)

    def embed_obs(self, obs):
        words = obs[:, :6]
        scores = obs[:, 6:]
        words = self.w_embed(words).flatten(1)
        scores = self.s_embed(scores).flatten(1)
        obs_emb = torch.cat([words, scores], dim=1)
        #print(words.shape, scores.shape, obs_emb.shape)
        return obs_emb 

    def step(self, obs):
        obs = self.embed_obs(obs)
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v.run_net(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

if __name__ == '__main__':

    env = WordleEnv()
    obs_dim = 6 + 6 * 5
    act_dim = env.n_actions
    hidden_dim = 32
    emb_dim = 32
    save_freq = 100

    obs_emb_dim = 6 * emb_dim + 6 * 5 * 3

    gamma = 0.99
    lamb = 0.95
    steps = 3000
    epochs = 30000
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    target_kl = 0.01

    ac = MLPActorCritic(obs_emb_dim, hidden_dim, act_dim, emb_dim)
    buf = PPOBuffer(obs_dim, size=steps, gamma=gamma, lamb=lamb)
    
    # Set up optimizers for policy and value function
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        print('Running epoch', epoch)
        ep_ret = 0
        for t in range(steps):
            a, v, logp = ac.step(torch.as_tensor(o[None], dtype=torch.long))

            next_o, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            
            # Update obs (critical!)
            o = next_o

            terminal = d
            epoch_ended = t == steps - 1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o[None], dtype=torch.long))
                    print('Total reward', ep_ret)
                else:
                    v = 0

                buf.finish_path(v)
                o, ep_len = env.reset(), 0

        data = buf.get()
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs = obs.long()
        ret = data['ret']

        # Train policy with multiple steps of gradient descent
        for i in range(80):
            pi_optimizer.zero_grad()

            # Policy loss
            pi, logp = ac.pi(ac.embed_obs(obs), act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            if approx_kl > 1.5 * target_kl:
                break

            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(80):
            vf_optimizer.zero_grad()
            loss_v = ((ac.v.run_net(ac.embed_obs(obs)) - ret)**2).mean()
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        print('Entropy', ent)
        print('Policy loss', loss_pi.item())
        print('Value loss', loss_v.item())

        if (epoch + 1) % save_freq == 0:
            torch.save(ac.state_dict(), 'wordle_agent.pth')
# ship_scheduler/agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import random
import numpy as np

from .config import SEED

def _init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

_init_seed(SEED)

def build_berth_features(state: dict) -> torch.Tensor:
    queue_list = state.get('waiting_ships', [])
    feats = []
    for entry in queue_list:
        d  = torch.tensor([entry['delay']], dtype=torch.float32)
        s  = torch.tensor([float(entry['storm_ahead'])], dtype=torch.float32)
        st = torch.tensor([entry['service_time']], dtype=torch.float32)
        feats.append(torch.cat([d, s, st]))
    if feats:
        return torch.stack(feats, dim=0)
    else:
        return torch.zeros((0,3), dtype=torch.float32)

class ActorCritic(nn.Module):
    def __init__(self, dep_in_dim, berth_feat_dim, hidden_dim=128):
        super().__init__()
        # departure body
        self.dep_net = nn.Sequential(
            nn.Linear(dep_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dep_mean    = nn.Linear(hidden_dim, 1)
        self.dep_log_std = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.dep_log_std.bias, -3.0)

        # berth head
        self.berth_mlp = nn.Sequential(
            nn.Linear(berth_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(dep_in_dim + berth_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, dep_feats: torch.Tensor, berth_feats: torch.Tensor):
        # departure distribution
        emb     = self.dep_net(dep_feats)
        mean    = self.dep_mean(emb).squeeze(-1)
        log_std = self.dep_log_std(emb).squeeze(-1).clamp(-10.0, 2.0)
        std     = torch.exp(log_std).clamp(1e-3, 1.0)
        mean    = mean.clamp(-1e3, 1e3)
        delay_dist = Normal(mean, std + 1e-6)

        # berth logits
        if berth_feats.size(0) > 0:
            logits = self.berth_mlp(berth_feats).squeeze(-1)
        else:
            logits = torch.zeros(0, device=dep_feats.device)

        # critic input
        if berth_feats.size(0) > 0:
            agg = berth_feats.mean(0)
            critic_in = torch.cat([dep_feats, agg], -1)
        else:
            critic_in = torch.cat([
                dep_feats,
                torch.zeros(berth_feats.size(1), device=dep_feats.device)
            ], -1)
        value = self.critic(critic_in).squeeze(-1)

        return delay_dist, logits, value

class A2CAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, update_interval=1):
        self.num_ships = len(env.ships)
        
        s0 = env.reset()
        dep0 = self.build_departure_features(s0).unsqueeze(0)
        berth0 = build_berth_features(s0)
        dep_dim   = dep0.size(1)
        berth_dim = berth0.size(1)

        self.net    = ActorCritic(dep_dim, berth_dim)
        self.opt    = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma  = gamma
        self.update_interval = update_interval

        # rollout storage
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.entropies = []

    def build_departure_features(self, state: dict) -> torch.Tensor:
        fq   = torch.tensor(state.get('future_queues',   []), dtype=torch.float32)
        fs   = torch.tensor(state.get('future_storms',   []), dtype=torch.float32)
        best = torch.tensor([ state.get('best_time',      0.0) ], dtype=torch.float32)
        gap  = torch.tensor([ state.get('gap_to_best',    0.0) ], dtype=torch.float32)
        pr   = torch.tensor([ state.get('progress_ratio', 0.0) ], dtype=torch.float32)
        sid_norm = torch.tensor([ state['ship_id'] / self.num_ships ], dtype=torch.float32)
        return torch.cat([fq, fs, best, gap, pr, sid_norm])

    def select_action(self, state: dict):
        dep_feats   = self.build_departure_features(state)
        berth_feats = build_berth_features(state)
        dist, logits, value = self.net(dep_feats, berth_feats)

        action = {}
        if state['decision_type'] == 'departure':
            d = dist.sample()
            action['delay'] = d.item()
            logp = dist.log_prob(d)
            ent  = dist.entropy()
        else:
            if logits.numel() > 0:
                probs = torch.softmax(logits, -1)
                m     = Categorical(probs)
                idx   = m.sample()
                action['berth_choice'] = idx.item()
                logp  = m.log_prob(idx)
                ent   = m.entropy()
            else:
                action['berth_choice'] = None
                logp = torch.tensor(0.0)
                ent  = torch.tensor(0.0)

        self.log_probs.append(logp)
        self.values.append(value)
        self.entropies.append(ent)
        return action

    def update(self, next_value, done):
        # build returns and advantages
        R = 0.0 if done else next_value.item()
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        values  = torch.stack(self.values)
        lps     = torch.stack(self.log_probs)
        ents    = torch.stack(self.entropies)

        advs = returns - values
        # normalise
        m, s = advs.mean(), advs.std(unbiased=False).clamp(min=1e-8)
        advs = (advs - m) / s

        actor_loss  = -(lps * advs.detach()).mean()
        critic_loss = advs.pow(2).mean()
        entropy_loss = -0.01 * ents.mean()

        loss = actor_loss + 0.5*critic_loss + entropy_loss
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()

        # clear buffers
        self.log_probs, self.values, self.rewards, self.entropies = [], [], [], []

    def train(self, env):
        # reset counters
        self.loop_count        = 0
        self.cumulative_reward = 0.0
        self.learning_curve    = []

        state = env.reset()
        step_count = 0
        while True:
            a, step_count = self.select_action(state), step_count+1
            next_s, r, done, _ = env.step(a)
            self.rewards.append(r)
            self.cumulative_reward += r

            if r != 0.0:
                self.loop_count += 1
                # record one more point
                self.learning_curve.append({
                    'loop': self.loop_count,
                    'cum_reward': self.cumulative_reward
                })

            # bootstrap next value
            depf = self.build_departure_features(next_s)
            berthf = build_berth_features(next_s)
            _, _, nv = self.net(depf, berthf)

            if r != 0.0 or (step_count % self.update_interval == 0):
                self.update(nv, done)

            if done:
                break
            state = next_s

        # flush remainder
        if self.rewards:
            self.update(torch.tensor(0.0), True)

        print("Training complete.")

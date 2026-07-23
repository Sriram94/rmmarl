from __future__ import annotations
import argparse
import numpy as np
import torch
from envs.overcooked import OvercookedEnv, N_ACTIONS
from envs.overcooked_tasks import build_overcooked_rms
from envs.sgrrm import SGRRM
from agents.factory import build_agent, ALGORITHMS

def run(args):
    device = 'cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu'
    env = OvercookedEnv(grid_size=args.grid_size, max_steps=args.max_steps, cook_time=args.cook_time, seed=args.seed)
    rms = build_overcooked_rms(env.agent_ids)
    sgrrm = SGRRM(env, rms)
    obs_dim = env.obs_dim
    shared_rm = rms[env.agent_ids[0]]
    u_to_index = {u: i for i, u in enumerate(shared_rm.states)}
    n_rm_states = shared_rm.num_states
    cooks = {aid: build_agent(args.algo, obs_dim=obs_dim, n_rm_states=n_rm_states, n_actions=N_ACTIONS, opponent_ids=[o for o in env.agent_ids if o != aid], n_opponent_actions=N_ACTIONS, device=device, seed=args.seed + i, buffer_size=args.buffer_size) for i, aid in enumerate(env.agent_ids)}
    epsilon = args.epsilon
    eps_min, eps_decay = (args.epsilon_min, args.epsilon_decay)
    episode_rewards = []
    for ep in range(args.episodes):
        obs = sgrrm.reset()
        prev_actions = {aid: 0 for aid in env.agent_ids}
        ep_reward = {aid: 0.0 for aid in sgrrm.agent_ids}
        deliveries = 0
        for t in range(args.max_steps):
            actions = {}
            for aid in env.agent_ids:
                s_vec, u = obs[aid]
                actions[aid] = cooks[aid].act(s_vec, u_to_index[u], epsilon)
            next_obs, rewards, dones, info = sgrrm.step(actions)
            if info['delivered']:
                deliveries += 1
            for aid in env.agent_ids:
                s_vec, u = obs[aid]
                s_next_vec, u_next = next_obs[aid]
                agent = cooks[aid]
                opp_prev = {o: prev_actions[o] for o in env.agent_ids if o != aid}
                opp_actions = {o: actions[o] for o in env.agent_ids if o != aid}
                agent.observe_opponents(s_vec, u_to_index[u], opp_actions, opp_prev)
                remember_kwargs = {}
                if agent.uses_counterfactual:
                    cf = sgrrm.counterfactual_targets(aid, sgrrm.env_state)
                    remember_kwargs.update(cf_targets=cf, u_to_index=lambda uu: u_to_index[uu])
                if agent.has_opponent_model:
                    remember_kwargs['opp_action'] = opp_actions
                agent.remember(s_vec, u_to_index[u], actions[aid], rewards[aid], s_next_vec, u_to_index[u_next], dones[aid], **remember_kwargs)
                agent.train(batch_size=args.batch_size)
            prev_actions = dict(actions)
            for aid in sgrrm.agent_ids:
                ep_reward[aid] += rewards[aid]
            obs = next_obs
            if any(dones.values()):
                break
        epsilon = max(eps_min, epsilon * eps_decay)
        team_total = sum((ep_reward[aid] for aid in env.agent_ids))
        episode_rewards.append(team_total)
        if (ep + 1) % max(1, args.log_every) == 0:
            avg = np.mean(episode_rewards[-args.log_every:])
            print(f'[{args.algo}] ep {ep + 1}/{args.episodes}  team_reward={team_total:.1f}  avg{args.log_every}={avg:.2f}  deliveries={deliveries}  eps={epsilon:.3f}')
    return episode_rewards

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=ALGORITHMS, default='dcrom')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--max_steps', type=int, default=12000)
    p.add_argument('--grid_size', type=int, default=15, help="Default 15 (bigger than the paper's ~7x7-ish Charakorn et al. layout; the paper doesn't pin an exact number here).")
    p.add_argument('--cook_time', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--log_every', type=int, default=5)
    p.add_argument('--buffer_size', type=int, default=200000, help='Replay buffer capacity per agent. Appendix I.4 specifies 2e7; default is reduced for memory tractability (see agents/factory.py docstring).')
    p.add_argument('--epsilon', type=float, default=0.9, help='Exploration rate. Appendix I.4 specifies a FIXED rate of 0.9 (the default); pass --epsilon_decay < 1.0 to opt into standard annealing instead.')
    p.add_argument('--epsilon_min', type=float, default=0.05)
    p.add_argument('--epsilon_decay', type=float, default=1.0, help='1.0 = no decay (matches the paper).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    return p
if __name__ == '__main__':
    args = build_argparser().parse_args()
    run(args)

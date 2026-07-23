from __future__ import annotations
import argparse
import numpy as np
import torch
from envs.overcooked import OvercookedEnv, N_ACTIONS
from envs.overcooked_tasks import build_overcooked_rms
from envs.sgrrm import SGRRM
from agents.centralized.factory import build_centralized_agent, CENTRALIZED_ALGORITHMS

def run(args):
    device = 'cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu'
    env = OvercookedEnv(grid_size=args.grid_size, max_steps=args.max_steps, cook_time=args.cook_time, seed=args.seed)
    rms = build_overcooked_rms(env.agent_ids)
    sgrrm = SGRRM(env, rms)
    obs_dim = env.obs_dim
    shared_rm = rms[env.agent_ids[0]]
    u_to_index = {u: i for i, u in enumerate(shared_rm.states)}
    n_rm_states = shared_rm.num_states
    team = build_centralized_agent(args.algo, agent_ids=env.agent_ids, obs_dim=obs_dim, n_rm_states=n_rm_states, n_actions=N_ACTIONS, device=device, seed=args.seed)
    epsilon = args.epsilon if not team.on_policy else 0.0
    eps_min, eps_decay = (args.epsilon_min, args.epsilon_decay)
    episode_rewards = []
    for ep in range(args.episodes):
        obs = sgrrm.reset()
        ep_reward = {aid: 0.0 for aid in sgrrm.agent_ids}
        deliveries = 0
        for t in range(args.max_steps):
            team_obs = {aid: obs[aid][0] for aid in env.agent_ids}
            team_u = {aid: u_to_index[obs[aid][1]] for aid in env.agent_ids}
            actions = team.act(team_obs, team_u, epsilon)
            next_obs, rewards, dones, info = sgrrm.step(actions)
            if info['delivered']:
                deliveries += 1
            next_team_obs = {aid: next_obs[aid][0] for aid in env.agent_ids}
            next_team_u = {aid: u_to_index[next_obs[aid][1]] for aid in env.agent_ids}
            team.store(team_obs, team_u, actions, rewards, next_team_obs, next_team_u, dones)
            team.train(batch_size=args.batch_size)
            for aid in sgrrm.agent_ids:
                ep_reward[aid] += rewards[aid]
            obs = next_obs
            if any(dones.values()):
                break
        if not team.on_policy:
            epsilon = max(eps_min, epsilon * eps_decay)
        team_total = sum((ep_reward[aid] for aid in env.agent_ids))
        episode_rewards.append(team_total)
        if (ep + 1) % max(1, args.log_every) == 0:
            avg = np.mean(episode_rewards[-args.log_every:])
            print(f'[{args.algo}] ep {ep + 1}/{args.episodes}  team_reward={team_total:.1f}  avg{args.log_every}={avg:.2f}  deliveries={deliveries}  eps={epsilon:.3f}')
    return episode_rewards

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=CENTRALIZED_ALGORITHMS, default='qmix')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--max_steps', type=int, default=12000)
    p.add_argument('--grid_size', type=int, default=15, help='Default 15.')
    p.add_argument('--cook_time', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--log_every', type=int, default=5)
    p.add_argument('--epsilon', type=float, default=0.9, help='Exploration rate for off-policy algos (MADDPG/QMIX). Appendix I.4 specifies a FIXED rate of 0.9 (the default); ignored for on-policy algos (MAPPO/self-play PPO), which explore via their stochastic policy.')
    p.add_argument('--epsilon_min', type=float, default=0.05)
    p.add_argument('--epsilon_decay', type=float, default=1.0, help='1.0 = no decay (matches the paper).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    return p
if __name__ == '__main__':
    args = build_argparser().parse_args()
    run(args)

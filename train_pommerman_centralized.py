from __future__ import annotations
import argparse
import numpy as np
import torch
from envs.pommerman import PommermanEnv, N_ACTIONS
from envs.pommerman_tasks import build_pommerman_rms
from envs.sgrrm import SGRRM
from agents.centralized.factory import build_centralized_agent, CENTRALIZED_ALGORITHMS
from agents.baselines import DQNBaseline

def run(args):
    device = 'cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu'
    env = PommermanEnv(n_per_team=args.n_per_team, grid_size=args.grid_size, min_steps=args.min_steps, max_steps_range=args.max_steps_range, bomb_timer=args.bomb_timer, blast_range=args.blast_range, seed=args.seed)
    rms = build_pommerman_rms(env)
    sgrrm = SGRRM(env, rms)
    team_ids = [aid for aid in env.agent_ids if env.team_of[aid] == 0]
    opp_ids = [aid for aid in env.agent_ids if env.team_of[aid] == 1]
    obs_dim = env.obs_dim
    team_rm = rms[team_ids[0]]
    u_to_index = {u: i for i, u in enumerate(team_rm.states)}
    n_rm_states = team_rm.num_states
    team = build_centralized_agent(args.algo, agent_ids=team_ids, obs_dim=obs_dim, n_rm_states=n_rm_states, n_actions=N_ACTIONS, device=device, seed=args.seed)
    opp_rm = rms[opp_ids[0]]
    opp_u_to_index = {u: i for i, u in enumerate(opp_rm.states)}
    opp_n_rm_states = opp_rm.num_states
    opponents = {aid: DQNBaseline(obs_dim, opp_n_rm_states, N_ACTIONS, device=device, seed=args.seed + 100 + i) for i, aid in enumerate(opp_ids)}
    epsilon = args.epsilon if not team.on_policy else 0.0
    eps_min, eps_decay = (args.epsilon_min, args.epsilon_decay)
    episode_rewards = []
    outcomes = {'WIN': 0, 'LOSE': 0, 'DRAW': 0}
    for ep in range(args.episodes):
        obs = sgrrm.reset()
        ep_reward = {aid: 0.0 for aid in sgrrm.agent_ids}
        for t in range(env.max_steps):
            team_obs = {aid: obs[aid][0] for aid in team_ids}
            team_u = {aid: u_to_index[obs[aid][1]] for aid in team_ids}
            team_actions = team.act(team_obs, team_u, epsilon)
            actions = dict(team_actions)
            for aid in opp_ids:
                s_vec, u = obs[aid]
                actions[aid] = opponents[aid].act(s_vec, opp_u_to_index[u], epsilon)
            next_obs, rewards, dones, info = sgrrm.step(actions)
            next_team_obs = {aid: next_obs[aid][0] for aid in team_ids}
            next_team_u = {aid: u_to_index[next_obs[aid][1]] for aid in team_ids}
            team_rewards = {aid: rewards[aid] for aid in team_ids}
            team_dones = {aid: dones[aid] for aid in team_ids}
            team.store(team_obs, team_u, team_actions, team_rewards, next_team_obs, next_team_u, team_dones)
            team.train(batch_size=args.batch_size)
            for aid in team_ids:
                if next_obs[aid][1] in ('WIN', 'LOSE', 'DRAW'):
                    outcomes[next_obs[aid][1]] += 1
            for aid in opp_ids:
                s_vec, u = obs[aid]
                s_next_vec, u_next = next_obs[aid]
                opponents[aid].remember(s_vec, opp_u_to_index[u], actions[aid], rewards[aid], s_next_vec, opp_u_to_index[u_next], dones[aid])
                opponents[aid].train(batch_size=args.batch_size)
            for aid in sgrrm.agent_ids:
                ep_reward[aid] += rewards[aid]
            obs = next_obs
            if any(dones.values()):
                break
        if not team.on_policy:
            epsilon = max(eps_min, epsilon * eps_decay)
        team_total = sum((ep_reward[aid] for aid in team_ids))
        episode_rewards.append(team_total)
        if (ep + 1) % max(1, args.log_every) == 0:
            avg = np.mean(episode_rewards[-args.log_every:])
            print(f'[{args.algo}] ep {ep + 1}/{args.episodes}  team_reward={team_total:.1f}  avg{args.log_every}={avg:.2f}  outcomes(cum)={outcomes}  eps={epsilon:.3f}  episode_max_steps={env.max_steps}')
    return episode_rewards

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=CENTRALIZED_ALGORITHMS, default='self_play_ppo')
    p.add_argument('--n_per_team', type=int, default=1, choices=[1, 2])
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--min_steps', type=int, default=800, help="Minimum per-episode step limit (Appendix I.3's stated value; default 800). Each episode independently samples its actual limit from [min_steps, min_steps + max_steps_range].")
    p.add_argument('--max_steps_range', type=int, default=200, help="Width of the per-episode step-limit sampling window above --min_steps (default 200, giving the paper's [800, 1000] range with the default --min_steps).")
    p.add_argument('--grid_size', type=int, default=15, help="Default 15. NOTE: the paper itself uses grid_size=11; 15 is bigger than the paper's setting, not a match to it. Pass --grid_size 11 to reproduce the paper's Pommerman board size.")
    p.add_argument('--bomb_timer', type=int, default=8)
    p.add_argument('--blast_range', type=int, default=2)
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

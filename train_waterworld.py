from __future__ import annotations
import argparse
import numpy as np
import torch
from envs.waterworld import WaterworldEnv, N_ACTIONS
from envs.waterworld_tasks import build_waterworld_rms, build_multitask_waterworld_rms
from envs.sgrrm import SGRRM
from agents.factory import build_agent, ALGORITHMS
from agents.dqn_agent import DQNAgent

def run(args):
    if args.max_steps is None:
        args.max_steps = 1500
    device = 'cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu'
    env = WaterworldEnv(max_steps=args.max_steps, seed=args.seed, n_pursuers=args.n_pursuers, n_poison=args.n_poison)
    if args.multitask:
        rms = build_multitask_waterworld_rms(n_pursuers=env.n_pursuers, n_poison=env.n_poison)
    else:
        rms = build_waterworld_rms(task_id=args.task, n_pursuers=env.n_pursuers, n_poison=env.n_poison)
    sgrrm = SGRRM(env, rms)
    pursuer_ids = [f'pursuer_{i}' for i in range(env.n_pursuers)]
    poison_ids = [f'poison_{i}' for i in range(env.n_poison)]
    obs_dim = env.obs_dim
    pursuer_rm = rms[pursuer_ids[0]]
    u_to_index = {u: i for i, u in enumerate(pursuer_rm.states)}
    n_rm_states = pursuer_rm.num_states
    pursuers = {pid: build_agent(args.algo, obs_dim=obs_dim, n_rm_states=n_rm_states, n_actions=N_ACTIONS, opponent_ids=poison_ids, n_opponent_actions=N_ACTIONS, device=device, seed=args.seed + i, buffer_size=args.buffer_size) for i, pid in enumerate(pursuer_ids)}
    poisons = {pid: DQNAgent(obs_dim, N_ACTIONS, device=device, seed=args.seed + 100 + i) for i, pid in enumerate(poison_ids)}
    epsilon = args.epsilon
    eps_min, eps_decay = (args.epsilon_min, args.epsilon_decay)
    episode_rewards = []
    for ep in range(args.episodes):
        obs = sgrrm.reset()
        prev_poison_actions = {pid: 0 for pid in poison_ids}
        ep_reward = {aid: 0.0 for aid in sgrrm.agent_ids}
        successes = 0
        for t in range(args.max_steps):
            actions = {}
            for pid in pursuer_ids:
                s_vec, u = obs[pid]
                actions[pid] = pursuers[pid].act(s_vec, u_to_index[u], epsilon)
            for pid in poison_ids:
                s_vec, _ = obs[pid]
                actions[pid] = poisons[pid].act(s_vec, epsilon)
            next_obs, rewards, dones, info = sgrrm.step(actions)
            for pid in pursuer_ids:
                s_vec, u = obs[pid]
                s_next_vec, u_next = next_obs[pid]
                agent = pursuers[pid]
                agent.observe_opponents(s_vec, u_to_index[u], actions, prev_poison_actions)
                remember_kwargs = {}
                if agent.uses_counterfactual:
                    cf = sgrrm.counterfactual_targets(pid, sgrrm.env_state)
                    remember_kwargs.update(cf_targets=cf, u_to_index=lambda uu: u_to_index[uu])
                if agent.has_opponent_model:
                    remember_kwargs['opp_action'] = {oid: actions[oid] for oid in poison_ids}
                agent.remember(s_vec, u_to_index[u], actions[pid], rewards[pid], s_next_vec, u_to_index[u_next], dones[pid], **remember_kwargs)
                agent.train(batch_size=args.batch_size)
                if rewards[pid] == 10.0:
                    successes += 1
            for pid in poison_ids:
                s_vec, _ = obs[pid]
                s_next_vec, _ = next_obs[pid]
                poisons[pid].store(s_vec, actions[pid], rewards[pid], s_next_vec, dones[pid])
                poisons[pid].train(batch_size=args.batch_size)
            prev_poison_actions = {pid: actions[pid] for pid in poison_ids}
            for aid in sgrrm.agent_ids:
                ep_reward[aid] += rewards[aid]
            obs = next_obs
            if any(dones.values()):
                break
        epsilon = max(eps_min, epsilon * eps_decay)
        pursuer_total = sum((ep_reward[pid] for pid in pursuer_ids))
        episode_rewards.append(pursuer_total)
        if (ep + 1) % max(1, args.log_every) == 0:
            avg = np.mean(episode_rewards[-args.log_every:])
            print(f'[{args.algo}] ep {ep + 1}/{args.episodes}  pursuer_reward={pursuer_total:.1f}  avg{args.log_every}={avg:.2f}  successes_this_ep={successes}  eps={epsilon:.3f}')
    return episode_rewards

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=ALGORITHMS, default='dcrom')
    p.add_argument('--task', type=int, default=1, choices=list(range(1, 7)), help='Which single Table-1 task to train on (1-6). Ignored if --multitask is set.')
    p.add_argument('--multitask', action='store_true', help='Cycle through all 6 Table-1 tasks per episode instead of training on one fixed task (Appendix I.1 / Section 7): +10 per completed task, looping 1->2->...->6->1, for as many completions as fit in --max_steps.')
    p.add_argument('--n_pursuers', type=int, default=3, help='Number of pursuer agents (default 3, matching the paper).')
    p.add_argument('--n_poison', type=int, default=2, help='Number of poison agents (default 2, matching the paper).')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--max_steps', type=int, default=None, help="Defaults to 1500, matching Appendix I.1/Section 7's episode budget (single-task and multitask alike -- single-task now loops the same task for repeated +10 rewards, same as multitask loops across tasks). Pass explicitly to override.")
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

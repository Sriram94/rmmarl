from __future__ import annotations
import argparse
import numpy as np
import torch
from envs.ant import TwoAgentAntEnv
from envs.ant_tasks import build_ant_rms, build_ant_state_reward_fns
from envs.sgrrm import SGRRM
from agents.continuous.factory import build_continuous_agent, CONTINUOUS_ALGORITHMS

def run(args):
    device = 'cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu'
    env = TwoAgentAntEnv(max_steps=args.max_steps, frame_skip=args.frame_skip, seed=args.seed)
    rms = build_ant_rms(env.agent_ids)
    shaping = build_ant_state_reward_fns(env)
    sgrrm = SGRRM(env, rms, state_reward_fns=shaping)
    obs_dim, action_dim = (env.obs_dim, env.action_dim)
    shared_rm = rms[env.agent_ids[0]]
    u_to_index = {u: i for i, u in enumerate(shared_rm.states)}
    n_rm_states = shared_rm.num_states
    agents = {aid: build_continuous_agent(args.algo, obs_dim=obs_dim, n_rm_states=n_rm_states, action_dim=action_dim, opponent_ids=[o for o in env.agent_ids if o != aid], n_opponent_actions=action_dim, device=device, seed=args.seed + i) for i, aid in enumerate(env.agent_ids)}
    epsilon = args.epsilon
    eps_min, eps_decay = (args.epsilon_min, args.epsilon_decay)
    episode_rewards = []
    for ep in range(args.episodes):
        obs = sgrrm.reset()
        prev_actions = {aid: np.zeros(action_dim, dtype=np.float32) for aid in env.agent_ids}
        ep_reward = {aid: 0.0 for aid in sgrrm.agent_ids}
        cycles = 0
        for t in range(args.max_steps):
            actions = {}
            for aid in env.agent_ids:
                s_vec, u = obs[aid]
                actions[aid] = agents[aid].act(s_vec, u_to_index[u], epsilon)
            next_obs, rewards, dones, info = sgrrm.step(actions)
            if rewards[env.agent_ids[0]] >= 1000:
                cycles += 1
            for aid in env.agent_ids:
                s_vec, u = obs[aid]
                s_next_vec, u_next = next_obs[aid]
                agent = agents[aid]
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
            print(f'[{args.algo}] ep {ep + 1}/{args.episodes}  team_reward={team_total:.1f}  avg{args.log_every}={avg:.2f}  cycles={cycles}  noise_scale={epsilon:.3f}  fell={env.state.fell}  final_x={env.state.torso_x:.2f}')
    return episode_rewards

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=CONTINUOUS_ALGORITHMS, default='dcrom')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--max_steps', type=int, default=2000)
    p.add_argument('--frame_skip', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--log_every', type=int, default=5)
    p.add_argument('--epsilon', type=float, default=1.0, help="Exploration-noise scale (continuous-action analogue of epsilon-greedy). Appendix I.4's 'fixed exploration rate' spec is for the discrete domains; here it defaults to a fixed (non-decayed) noise scale for consistency.")
    p.add_argument('--epsilon_min', type=float, default=0.05)
    p.add_argument('--epsilon_decay', type=float, default=1.0, help='1.0 = no decay.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    return p
if __name__ == '__main__':
    args = build_argparser().parse_args()
    run(args)

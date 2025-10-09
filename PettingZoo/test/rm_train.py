# train_waterworld_rm_pettingzoo.py

from waterworld_rm_env import env
from your_rm_agents import DQNCrossProductAgent, DCROMAgent  # etc
from reward_machine import RewardMachine  # your RM logic

def label_fn(env_obs_all: Dict[str, np.ndarray], agent_idx: int) -> Any:
    # Example: if agent sees any food within its sensor, event = 1 else 0
    obs = env_obs_all["agent_" + str(agent_idx)]
    # obs encodes relative vectors to foods; if any non-zero entry, pick event=1
    if np.any(obs != 0.0):
        return 1
    else:
        return 0

def build_rm():
    # sample RM: two states {0,1}, with reward when event occurs
    delta = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1,
    }
    sigma = {
        (0, 1): 1.0,
        (1, 1): 0.0,
    }
    return RewardMachine(n_states=2, delta=delta, sigma=sigma, label_fn=label_fn, terminal_states=[1])

def train(n_episodes=200, max_steps=100):
    env_instance = env()
    for ep in range(n_episodes):
        obs = env_instance.reset()
        # RM states for each agent
        rm_states = {ag: 0 for ag in env_instance.agents}

        for agent in env_instance.agents:
            # instantiate agent
            rm = build_rm()
            dim = obs[agent].shape[0]
            # action count = env_instance.action_spaces[agent].n
            myagent = DQNCrossProductAgent(obs_dim=dim, n_actions=env_instance.action_spaces[agent].n, rm=rm)
            # store in dict
            # (you might want to reuse or share initial agents across episodes)
        
        for step in range(max_steps):
            for agent in env_instance.agent_iter():
                ob = env_instance.observe(agent)
                u = rm_states[agent]
                a = myagent.select_action(ob, u)
                env_instance.step(a)
                # after step, env_instance._cumulative_rewards updated
                reward = env_instance._cumulative_rewards[agent]
                done = env_instance._dones[agent]
                # get next observation
                ob_next = env_instance.observe(agent)
                u_next = myagent.rm.next_state(u, obs)  # or label_fn wrapper
                r_rm = myagent.rm.reward(u, u_next)
                myagent.store_transition(ob, u, a, r_rm, ob_next, u_next, done)
                myagent.train_step()
                rm_states[agent] = u_next

        print("Episode", ep, "done")

if __name__ == "__main__":
    train()

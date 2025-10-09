import numpy as np
import random
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_state import OvercookedState

# Import your existing DQNCrossProductAgent
from DQN import DQNCrossProductAgent


# ============================================================
# Reward Machine Definition
# ============================================================
class RewardMachine:
    def __init__(self):
        self.state = "s0"
        self.ingredient_count = 0

    def reset(self):
        self.state = "s0"
        self.ingredient_count = 0

    def step(self, event):
        reward = 0.0

        if self.state == "s0":
            if event in ["onion_added", "tomato_added", "carrot_added"]:
                self.ingredient_count += 1
                if self.ingredient_count >= 3:
                    self.state = "s1"
                    self.ingredient_count = 0

        elif self.state == "s1":
            if event == "soup_cooked":
                self.state = "s2"

        elif self.state == "s2":
            if event == "soup_delivered":
                reward = 100.0
                self.state = "s0"  # reset for next soup

        return reward


def make_overcooked_env(layout_name="cramped_room", horizon=1200):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    return env


def encode_state(obs):
    """Flatten the joint observation into a vector."""
    return np.concatenate([obs["both_agent_obs"].flatten()])


def extract_event(prev_state: OvercookedState, next_state: OvercookedState):
    """
    Derives high-level symbolic events from true OvercookedState transitions.
    Compatible with overcooked_ai_py>=1.1.0.
    """

    # ---- Check pot states for ingredient additions ----
    for loc, pot in next_state.pot_states.items():
        prev_pot = prev_state.pot_states.get(loc, None)
        if pot["num_ing"] > (prev_pot["num_ing"] if prev_pot else 0):
            # Ingredient added
            ing_types = ["onion", "tomato", "carrot"]
            # Randomly assign type if multiple ingredients (toy heuristic)
            return random.choice([f"{t}_added" for t in ing_types])

        # Soup cooked
        if pot["cooking_tick"] == 0 and prev_pot and prev_pot["cooking_tick"] > 0:
            return "soup_cooked"

    # ---- Check deliveries ----
    if next_state.num_delivered > prev_state.num_delivered:
        return "soup_delivered"

    # ---- Check player holding state ----
    for i, player in enumerate(next_state.players):
        prev_player = prev_state.players[i]
        if player.has_object() and not prev_player.has_object():
            held_obj = player.get_object().name
            if held_obj in ["onion", "tomato", "carrot"]:
                return f"{held_obj}_picked"

    return None


def train_two_crossproduct_agents_with_rm(num_episodes=1000, target_update_freq=50):
    env = make_overcooked_env("cramped_room")
    obs = env.reset()
    state_dim = encode_state(obs).shape[0]
    n_actions = len(env.mdp.get_actions())

    # Two cooperative DQN agents
    agent1 = DQNCrossProductAgent(state_dim, n_actions)
    agent2 = DQNCrossProductAgent(state_dim, n_actions)

    rm1, rm2 = RewardMachine(), RewardMachine()

    for episode in range(num_episodes):
        env_state = env.mdp.get_standard_start_state()
        rm1.reset()
        rm2.reset()
        done = False
        step_count = 0
        total_r1, total_r2 = 0.0, 0.0

        while not done and step_count < 1200:
            obs = env.lossless_state_encoding(env_state)
            s1 = encode_state(obs)

            a1 = agent1.select_action(s1, rm1.state)
            a2 = agent2.select_action(s1, rm2.state)
            joint_action = (a1, a2)

            next_state, base_rewards, done, info = env.step(joint_action)

            # Extract symbolic event
            event = extract_event(env_state, next_state)
            currentrm1state = rm1.state
            currentrm2state = rm2.state
            r1 = rm1.step(event) if event else 0.0
            r2 = rm2.step(event) if event else 0.0
            newrm1state = rm1.state
            newrm2state = rm2.state

            s2 = encode_state(env.lossless_state_encoding(next_state))

            # Store transition and learn
            agent1.store_transition(s1, currentrm1state, a1, r1, s2, newrm1state, done)
            agent2.store_transition(s1, currentrm2state, a2, r2, s2, newrm2state, done)
            agent1.learn()
            agent2.learn()

            env_state = next_state
            total_r1 += r1
            total_r2 += r2
            step_count += 1

        if episode % target_update_freq == 0:
            agent1.update_target_network()
            agent2.update_target_network()

        print(f"[Episode {episode}] Steps={step_count} | Agent1={total_r1:.1f}, Agent2={total_r2:.1f}")


if __name__ == "__main__":
    train_two_crossproduct_agents_with_rm()

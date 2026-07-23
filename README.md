# Multi-Agent Reinforcement Learning with Reward Machines for Mixed Cooperative-Competitive Environments

Code base for the RLC 2026 paper: Multi-Agent Reinforcement Learning with Reward Machines for Mixed Cooperative-Competitive Environments.
 

This codebase implements:
- The core Reward Machine (RM) / Stochastic Game with RM rewards (SGRRM) abstractions
- CROM and QROM, with counterfactual
  experience generation and opponent modelling
- All baselines: DQN, DQN-OM, MA-DQN, DQRM, DHRL, DHRL-RM
- All centralized baselines: MADDPG, MAPPO, self-play PPO, QMIX
- All continuous-action variants: DCROM, DQROM, DDPG, DDPG-OM,
  MA-DDPG, DQRM, DHRL, DHRL-RM
- Four environments: Waterworld, Overcooked, Pommerman (1v1 and 2v2), and the
  two-agent MuJoCo Ant

---

## 1. Requirements

- **Python 3.9+** (developed/tested on 3.12)
- **PyTorch** (CPU is enough to run everything; GPU is used automatically if available)
- **NumPy**
- **MuJoCo** (only required for the Ant environment, `train_ant.py`)

No other environment needs external assets, simulators, or game engines, Waterworld, Overcooked, and Pommerman are all self-contained Python/NumPy
implementations in `envs/`.

### Installing

```bash
# Core dependencies (needed for every environment except Ant)
pip install torch numpy

# Only needed for the Ant environment
pip install mujoco
```

If you're on a system that restricts global pip installs (e.g. Debian/Ubuntu
with PEP 668), add `--break-system-packages`, or use a virtual environment
instead:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch numpy mujoco
```

**Notes on MuJoCo:** the `mujoco` PyPI package (v2.3+) ships prebuilt
binaries and runs headless out of the box. No separate MuJoCo license,
download, or `mujoco-py`/`dm_control` install is needed, and no display/X
server is required since we only step physics (no rendering). 

**Versions this was developed against** (anything reasonably recent should work):
```
Python 3.12.3
torch   2.13.0
numpy   2.4.4
mujoco  3.10.0
```

### Verifying your install

```bash
cd rmmarl
python3 -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"
python3 -c "import mujoco; print('mujoco', mujoco.__version__)"   # only if you installed it
```

---

## 2. Project layout

```
rmmarl/
  envs/                          # environments + reward machines
    reward_machine.py            # core RM class
    sgrrm.py                     # SGRRM wrapper + non-simple-RM hook
    waterworld.py, waterworld_tasks.py
    overcooked.py, overcooked_tasks.py
    pommerman.py, pommerman_tasks.py
    ant.py, ant_tasks.py, assets/two_agent_ant.xml
  agents/                        # discrete-action algorithms
    base.py                      # common Agent interface
    crom_agent.py                # DCROM, DQROM
    baselines.py                 # DQN, DQN-OM, MA-DQN, DQRM
    hierarchical.py              # DHRL, DHRL-RM
    dqn_agent.py                 # plain DQN 
    factory.py                   # build_agent(algo, ...)
    centralized/                 # Team-based baselines
      maddpg.py, mappo.py, qmix.py, self_play_ppo.py, factory.py
    continuous/                  # Continuous-action algorithms
      ddpg_agent.py              # DCROM, DQROM (continuous)
      baselines.py                # DDPG, DDPG-OM, MA-DDPG, DQRM (continuous)
      hierarchical.py             # DHRL, DHRL-RM (continuous)
      factory.py                  # build_continuous_agent(algo, ...)
  train_waterworld.py             # independent-agent training, any discrete algo
  train_waterworld_centralized.py # centralized team training (MADDPG/MAPPO/QMIX/self-play PPO)
  train_overcooked.py / train_overcooked_centralized.py
  train_pommerman.py / train_pommerman_centralized.py
  train_ant.py                    # continuous-action training
```

Every training script is self-contained: run it directly, no separate
"install the package" step needed beyond the pip installs above (there are
no `setup.py`/`pyproject.toml`, just run scripts from inside the
`rmmarl/` directory so the `envs`/`agents` imports resolve).

---

## 3. Running the discrete-action environments (Waterworld, Overcooked, Pommerman)

These three environments only need `torch` + `numpy`.

### 3.1 Waterworld

Independent-agent algorithms (pursuers vs. an independent-DQN poison team):

```bash
cd rmmarl
python3 train_waterworld.py --algo dcrom --task 1 --episodes 200
```

`--algo` choices: `dcrom, dqrom, dqn, dqn_om, ma_dqn, dqrm, dhrl, dhrl_rm`
`--task` chooses one of the 6 tasks (`1`-`6`, e.g. `1` = "red then green").



**Multi-task mode**: instead of repeating one fixed task, the pursuers cycle through all 6 Table-1 tasks in order within
a single (also 1500-step-by-default) episode, earning +10 for each
completed task and looping back to task 1 after task 6:

```bash
python3 train_waterworld.py --algo dcrom --multitask --episodes 200
```

`--multitask` overrides `--task` (which is ignored if both are passed).
Both single-task and multitask modes default `--max_steps` to 1500,
matching the paper's episode budget; pass `--max_steps` explicitly to
override.



Centralized team baselines as the pursuer team:

```bash
python3 train_waterworld_centralized.py --algo maddpg --task 1 --episodes 200
```


`--algo` choices: `maddpg, mappo, self_play_ppo, qmix`

Other useful flags (all scripts): `--max_steps`, `--batch_size`, `--log_every`,
`--seed`, `--cpu` (force CPU even if a GPU is available).


**Multi-task mode**


```bash
python3 train_waterworld_centralized.py --algo maddpg --multitask --episodes 200
```



### 3.2 Overcooked

```bash
python3 train_overcooked.py --algo dqrom --episodes 200
python3 train_overcooked_centralized.py --algo qmix --episodes 200   
```

Extra flags: `--grid_size` (default 15), `--cook_time` (default 10).

### 3.3 Pommerman

```bash
python3 train_pommerman.py --algo dcrom --n_per_team 1 --episodes 200       # 1v1
python3 train_pommerman.py --algo dqrom --n_per_team 2 --episodes 200       # 2v2
python3 train_pommerman_centralized.py --algo self_play_ppo --n_per_team 2 --episodes 200
```

`--n_per_team 1` = 1v1, `--n_per_team 2` = 2v2. Extra flags: `--grid_size`
(default 15), `--bomb_timer`, `--blast_range`.



**Episode length is variable**. Rather than a single fixed step limit, each episode independently samples its own limit from `[min_steps, min_steps + max_steps_range]` (default `[800, 1000]`), resampled fresh on every `reset()`. `--min_steps` defaults to 800 but is fully user-settable; `--max_steps_range` controls the width of the window above it (default 200). 


```bash
python3 train_pommerman.py --algo dcrom --min_steps 800 --episodes 200                    
python3 train_pommerman.py --algo dcrom --min_steps 500 --episodes 200                    # shorter episodes
python3 train_pommerman.py --algo dcrom --min_steps 800 --max_steps_range 400 --episodes 200  # wider [800, 1200] window
```

Each logged line reports `episode_max_steps`, the actual step limit sampled
for that episode, so you can see the variation directly during training.


---

## 4. Running the Ant environment

Requires `mujoco` in addition to `torch`/`numpy` (see Section 1).

```bash
cd rmmarl
python3 train_ant.py --algo dcrom --episodes 200
```

`--algo` choices: `dcrom, dqrom, ddpg, ddpg_om, ma_ddpg, dqrm, dhrl, dhrl_rm`

Extra flags: `--max_steps` (default 2000), `--frame_skip` (MuJoCo substeps per env step, default 5).

If `python3 train_ant.py ...` fails with a MuJoCo import or GL/EGL error,
first confirm `python3 -c "import mujoco; mujoco.MjModel.from_xml_path('envs/assets/two_agent_ant.xml')"`
runs cleanly on its own, this only exercises physics stepping (no
rendering), so it should work on any headless machine once `pip install
mujoco` succeeds.

---

## 5. Quick sanity checks

To confirm everything is wired correctly without waiting for a full training
run, use a small number of episodes/steps -- every script accepts
`--episodes` and `--max_steps`:

```bash
# ~10 seconds each, just confirms no errors and prints per-episode logs
python3 train_waterworld.py --algo dcrom --episodes 3 --max_steps 60 --log_every 3 --cpu
python3 train_overcooked.py --algo dqrom --episodes 3 --max_steps 80 --log_every 3 --cpu
python3 train_pommerman.py --algo dqn --n_per_team 1 --episodes 3 --max_steps 800 --log_every 3 --cpu
python3 train_ant.py --algo ddpg --episodes 2 --max_steps 100 --log_every 2 --cpu
```

For meaningful learning curves, you will want many more episodes (the paper trains for thousands of episodes
per domain) and likely a GPU for the deep variants with larger networks
(DCROM's cross-product network in particular).

---


## 6. Hyperparameters

Defaults for all algorithms match the hyperparameters for the experiments in the paper, with one exception. The default for the replay buffer size is left to 200000 while the paper uses 20000000. This is done so that the code can run off-the-shelf on resource constrained environments. 

---



## 7. Adding a new algorithm or environment

- **New discrete algorithm**: implement the `Agent` interface in
  `agents/base.py` (`act`, `observe_opponents`, `remember`, `train`), then
  register it in `agents/factory.py`. Every existing training script will
  pick it up automatically via `--algo`.
- **New centralized (team) algorithm**: implement `CentralizedController`
  in `agents/centralized/base.py`, register in `agents/centralized/factory.py`.
- **New continuous-action algorithm**: same pattern under `agents/continuous/`.
- **New environment**: implement `reset()` / `step(actions)` / `observe(agent_id, state)`
  / `agent_ids` (see any file in `envs/` for the exact shape expected), write
  a matching `*_tasks.py` with `RewardMachine` objects and a labelling
  function, wrap in `SGRRM`, and copy one of the `train_*.py` scripts as a
  template, none of the algorithm code needs to change.

---

## 8. Note

This is research code and will not be actively maintained. Please send an email to ***sriramsubramanian@cunet.carleton.ca*** for questions or comments.

---

## 9. Code Citations

We would like to cite [Playground](https://github.com/MultiAgentLearning/playground) for the game design and mechanics that our Pommerman environment implementation is based on.

We would like to cite [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) for the game design and mechanics that our Waterworld environment implementation is based on. 

We would like to cite [Overcooked](https://github.com/HumanCompatibleAI/overcooked_ai) for the game design and mechanics that our Overcooked environment implementation is based on. 

We would like to cite [MuJoCo](https://github.com/google-deepmind/mujoco) for the game design and mechanics that our Ant environment implementation is based on. 

All of the above mentioned repositories have been released under open (MIT or Apache2) licenses. 

---


## Paper citation

If you found this helpful, please cite the following paper:

<pre>

@article{Sriramrmmarl2026,
  title = 	 {Multi-Agent Reinforcement Learning with Reward Machines for Mixed Cooperative-Competitive Environments},
  author = 	 {Subramanian, Sriram Ganapathi and Klassen, Toryn and McIlraith, Sheila} 
  journal =  {Reinforcement Learning Journal},
  year = 	 {2026}
}

</pre>





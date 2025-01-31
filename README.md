# Multi-Agent Reinforcement Learning with Reward Machines for Mixed Cooperative-Competitive Environments

Code base for the ICML 2025 submission (Submission Number 10396): Multi-Agent Reinforcement Learning with Reward Machines for Mixed Cooperative-Competitive Environments.

Note: This is a restricted version due to file size, licensing, and anonymity considerations. Full data and code will be
open-sourced with the paper.
 
## Code structure


- See folder playground for Pommerman environment.

- See folder Gymnasium-Robotics for multi-agent Ant environment.

- See folder PettingZoo  for multi-agent Waterworld environment.

- See folder overcooked_ai for the Overcooked environment.



## Installation Instructions for Ubuntu 18.04




##### Requirements

Atleast 

- `python==3.7.11`


```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```


- `Tkinter`

```shell
sudo apt-get update
sudo apt-get install python3-tk
```


- `tensorflow 2`

```shell
pip install --upgrade pip
pip install tensorflow
```

- `pandas`

```shell
pip install pandas
```
- `matplotlib`

```shell
pip install matplotlib
```

Download the files and store them in a separate directory to install packages from the requirements file. 

```shell
cd playground
pip install -U . 
```


For more help with the installation, look at the instrctions in [Playground](https://github.com/MultiAgentLearning/playground). 

Now you can just run the respective files mentioned in the above section to run our code.


For the Pursuit domain you also need to install petting zoo library. 

### Petting Zoo (Waterworld Environment)


##### Requirements

Atleast

- `Gym` (Version 0.18.0)

```shell
pip install gym==0.18.0
```

- `pettingzoo` (Version 1.14.0) 

```shell
cd pettingzoo 
pip install ./
```


Please install petting zoo using the instruction above and not from source (our files contains some differences from the source files). 

Now, you can just run the relevant files mentioned in the above section to run our code. 


For installation of Overcooked follow the instructions [here](https://github.com/HumanCompatibleAI/overcooked_ai), and
for the installation of Multi-agent Mujoco follow the instructions [here](https://robotics.farama.org/content/installation/)



# Multi-Agent Reinforcement Learning with Reward Machines for Mixed Cooperative-Competitive Environments

Code base for the RLC 2026 submission (Submission Number 91): Multi-Agent Reinforcement Learning with Reward Machines for Mixed Cooperative-Competitive Environments.

Note: This is a restricted version due to file size, licensing, and anonymity considerations. Full data and code will be
open-sourced with the paper.
 
## Code structure


- See folder algorithms for all our algorithmic implementations.

- See folder playground for Pommerman environment.

- See folder PettingZoo for Multi-Agent Waterworld environment.

- See folder overcooked\_ai for the Overcooked environment.



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

Now, we are ready to start training. This provides an example with a two agent version of Pommerman with one agent playing DCROM and the other agent playing DQN. The file can be modified with our algorithm scripts for all other training and execution performances. 


```shell
cd playground/pommerman
python DCROMvsDQN.py
```


Now you can just run the respective files mentioned in the above section to run our code. Use n\_agents=2 for the two agent version and n\_agents=4 for the team version of Pommerman.


For the Waterworld domain you also need to install petting zoo library. 

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




Now, we are ready to start training. This provides an example with Pursuer agents playing DCROM and Poison agents playing DQN. The file can be modified with our algorithm scripts for all other training and execution performances. 


```shell
cd PettingZoo/test
python waterworld.py
```


Now you can just run the respective files mentioned in the above section to run our code.


### Overcooked


```shell
cd overcooked_ai
uv venv
uv sync
```

To verify installation, 

```shell
python testing/overcooked_test.py
```


Please install Overcooked using the instruction above and not from source (our files contains some differences from the source files). For installation help, more details are available [here](https://github.com/HumanCompatibleAI/overcooked_ai).




Now, we are ready to start training. This provides an example with both Overcooked agents playing DQN. The DQN agents can be replaced with other agents in the algorithm folder for other experiments. 


```shell
cd overcooked_ai
python overcooked_training.py
```

## Code Citations

We would like to cite [Playground](https://github.com/MultiAgentLearning/playground) for code providing the environments used in the Pommerman experiments. 

We would like to cite [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) for code providing the environments used in the Waterworld experiments. 

We would like to cite [Overcooked](https://github.com/HumanCompatibleAI/overcooked_ai) for code providing the environments used in the Overcooked experiments. 

All of the above mentioned repositories have been released under open (MIT or Apache2) licenses. 



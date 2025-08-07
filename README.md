# Deep-Reinforcement-Learning-Project

**"Deep RL Only"**: contain code and graphs for code that use DDPG, SAC, and PPO for the Franka robot arm
1. "franka.py": define the environment for the Franka robot arm to be passed into Stable Baseline 3.
2. "training.py": this is where the training and evaluation happen.

**"Proportional Derivative Control + Deep RL"**: contain Jupyter notebooks for code and graph that use DDPG, SAC, and PPO with Proportional Derivative Controller for the Franka robot arm
1. PD_Control.ipynb shows the PD controller in RL training structure.
2. RL_PD_Control.ipynb adds the RL action in tau to compensate the external force.
3. Result_Comparison.ipynb compares the PD result and PD-RL result.

**"requirements.txt"**: define the require dependency to run everything in this repo.
1. Make sure Python 3.11 is install on your machine.
2. Create a Python virtual env with 
```python 
python3.11 -m venv .venv
```

3. Activate the environement
```python
source .venv/bin/activate
```

4. Install the necessary libraries with
```python
pip install -r requirements.txt
```

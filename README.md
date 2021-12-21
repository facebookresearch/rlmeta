# RLMeta

rlmeta - a flexible lightweight research framework for Distributed Reinforcement Learning

## Installation
To build from source, please install [`PyTorch`](https://pytorch.org/) first, and then run the commands below
```
git submodule sync && git submodule update --init --recursive
pip install -e .
```

## Run an Example
To run the example for Atari Pong game with PPO algorithm:
```
cd examples/atari/ppo
python atari_ppo.py
```
The logs and checkpoints will be automatically saved to examples/atari/ppo/outputs/{yyyy-mm-dd}/{hh-mm-ss}/

## License
rlmeta is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Frozen Evo-MAML

In this repository you can find the implementations of 2 Meta Learning algorithms tuned for Meta Reinforcement Learning (Meta LR):
- Standard Model-Agnostic Meta-Learning [(MAML)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://arxiv.org/abs/1703.03400&ved=2ahUKEwjI_afgruiQAxXbGxAIHSaBNvMQFnoECBoQAQ&usg=AOvVaw3r3ZhC-vf1gt9e7gyFUG0W) algorithm;
- Frozen Evo-MAML algorithm: improvement of the original MAML with the use of layer freezing and Evolutionary Techniques. 

Detailed description of Frozen Evo-MAML can be found [here](https://drive.google.com/file/d/1Q_3dSOD2XRnToLJeNT80pbpRErm1MVMT/view?usp=sharing).

## Repository structure

The repository contains 2 main folders:
- [`maml`](https://github.com/RelentlessViper/frozen-evo-maml/tree/main/maml) contains the core implementations of MAML and Frozen Evo-MAML as well as task distribution based on [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and basic training and logging functions.
- [`notebooks`](https://github.com/RelentlessViper/frozen-evo-maml/tree/main/notebooks) contains an example of training the classic MAML algorithm (Frozen Evo-MAML can be trained in the same way).

## Installation

```bash
git clone https://github.com/RelentlessViper/frozen-evo-maml.git
cd frozen-evo-maml
pip install -r requirements.txt
```

## Example use:

```python
# Import necessary dependencies
import sys
import os
from maml import train_multiple_mani

# Set up different training settings
train_settings = {
    "conservative": {
        "num_tasks": 20,
        "num_inner_steps": 1,
        "meta_batch_size": 10,
        "inner_lr": 0.01,
        "outer_lr": 0.001,
        "num_episodes": 20,
        "eval_fr": 5,
        "algorithm_type": "maml",
    },
    "balanced": {
        "num_tasks": 30,
        "num_inner_steps": 5,
        "meta_batch_size": 10,
        "inner_lr": 0.01,
        "outer_lr": 0.001,
        "num_episodes": 15,
        "eval_fr": 3,
        "algorithm_type": "frozen-evo-maml",
    },
}

# Launch training
results, train_times = train_multiple_mani(train_settings, save_path="train_results_classic")
print(results)
```

The training and testing metrics will be saved in 2 files named as follows:
- The file named `<save_path>` + `.csv` will contain all the metrics (losses and average rewards).
- The file named `<save_path>` + `_time.csv` will contain the time spent training models in given settings.

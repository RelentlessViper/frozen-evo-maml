import pandas as pd
import time
from .modified_cartpole import CartPoleTaskDistribution
from .classic_maml import MAML
from .frozen_evo_maml import FrozenEvoMAML

def train_maml(
    num_tasks: int = 50,
    num_inner_steps: int = 1,
    meta_batch_size: int = 5,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    num_episodes: int = 20,
    eval_fr: int = 5,
    algorithm_type: str = "maml",
) -> pd.DataFrame:
    """
    Train a meta-learning model with specified hyperparameters and algorithm type.
    
    This function initializes a task distribution and the selected meta-learning
    algorithm, then runs the training process for the specified number of episodes.
    Supports both standard MAML and Frozen Evo-MAML algorithms.
    
    Parameters
    ----------
    num_tasks : int, optional
        Total number of tasks in the task distribution, by default 50
    num_inner_steps : int, optional
        Number of gradient steps in the inner loop, by default 1
    meta_batch_size : int, optional
        Number of tasks per meta-update, by default 5
    inner_lr : float, optional
        Learning rate for inner loop adaptation, by default 0.01
    outer_lr : float, optional
        Learning rate for meta-optimizer, by default 0.001
    num_episodes : int, optional
        Total number of training iterations, by default 20
    eval_fr : int, optional
        Evaluation frequency during training, by default 5
    algorithm_type : str, optional
        Type of meta-learning algorithm to use. Must be either 'maml' for 
        standard MAML or 'frozen-evo-maml' for Frozen Evolutionary MAML, 
        by default 'maml'
        
    Returns
    -------
    Tuple[pd.DataFrame, float]
        A tuple containing:
        - DataFrame with training results and metrics
        - Total training time in seconds
        
    Raises
    ------
    Exception
        If an unsupported algorithm_type is provided
    """
    task_dist = CartPoleTaskDistribution(num_tasks=num_tasks)
    
    if algorithm_type == "maml":
        maml = MAML(
            task_distribution=task_dist, 
            inner_lr=inner_lr,
            meta_lr=outer_lr,
            num_inner_steps=num_inner_steps,
            eval_fr=eval_fr,
        )
    elif algorithm_type == "frozen-evo-maml":
        maml = FrozenEvoMAML(
            task_distribution=task_dist, 
            inner_lr=inner_lr,
            meta_lr=outer_lr,
            num_inner_steps=num_inner_steps,
            eval_fr=eval_fr,
        )
    else:
        raise Exception("Only 2 algorithms can be selected: 'maml' or 'frozen-evo-maml'")

    start_time = time.time()
    results = maml.train(num_iterations=num_episodes, meta_batch_size=meta_batch_size)
    end_time = time.time()
    return (results, end_time - start_time)

def train_multiple_maml(
    train_settings: dict[str, dict[str, int | float]],
    save: bool = True,
    save_path: str = "train_results",
) -> pd.DataFrame:
    """
    Train multiple meta-learning models with different hyperparameter settings.
    
    This function runs multiple training experiments with varying hyperparameters
    and algorithm types, then aggregates the results for comparison and analysis.
    
    Parameters
    ----------
    train_settings : Dict[str, Dict[str, int | float | str]]
        Dictionary where keys are setting names and values are dictionaries
        containing hyperparameter configurations. Expected hyperparameters:
        - num_tasks: Total number of tasks
        - num_inner_steps: Inner loop gradient steps
        - meta_batch_size: Tasks per meta-update
        - inner_lr: Inner learning rate
        - outer_lr: Outer learning rate
        - num_episodes: Training iterations
        - eval_fr: Evaluation frequency
        - algorithm_type: Type of meta-learning algorithm ('maml' or 'frozen-evo-maml')
    save : bool, optional
        Whether to save results to CSV files, by default True
    save_path : str, optional
        Base path for saving results, by default "train_results"
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing:
        - DataFrame with aggregated training results from all settings
        - Series containing training times for each setting
    """
    results = pd.DataFrame(
        data={
            "episode": [],
            "setting_name": [],
            "n-way/m-shot": [],
            "meta_loss": [],
            "meta_rewards": [],
            "test_loss": [],
            "test_rewards": [],
        }
    )
    train_times = []
    for train_setting_name, train_setting_dict in train_settings.items():
        print(f"Current train setting: {train_setting_name}")
        current_result, current_time = train_maml(
            num_tasks=train_setting_dict.get("num_tasks"),
            num_inner_steps=train_setting_dict.get("num_inner_steps"),
            meta_batch_size=train_setting_dict.get("meta_batch_size"),
            inner_lr=train_setting_dict.get("inner_lr"),
            outer_lr=train_setting_dict.get("outer_lr"),
            num_episodes=train_setting_dict.get("num_episodes"),
            eval_fr=train_setting_dict.get("eval_fr"),
            algorithm_type=train_setting_dict.get("algorithm_type"),
        )
        train_times.append(current_time)
        current_result["setting_name"] = train_setting_name
        results = pd.concat(
            [
                results,
                current_result,
            ],
            axis=0,
        )

    results = results.astype({"episode": "int32"}).reset_index(drop=True)
    train_times = pd.Series(
        train_times,
    )
    if save:
        results.to_csv(save_path + ".csv", index=False)
        train_times.to_csv(save_path + "_time.csv", index=False)

    return (results, train_times)
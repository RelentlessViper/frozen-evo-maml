from gymnasium.envs.classic_control.cartpole import CartPoleEnv

import numpy as np
from typing import Dict, List

class ModifiedCartPoleEnv(CartPoleEnv):
    """
    A modified version of the CartPole environment with customizable physical parameters.
    
    This environment extends the standard CartPole environment to allow dynamic
    modification of physical parameters, enabling the creation of diverse tasks
    for meta-learning and domain adaptation scenarios.
    
    Attributes
    ----------
    masscart : float
        Mass of the cart in kilograms
    masspole : float
        Mass of the pole in kilograms  
    length : float
        Half-length of the pole in meters
    force_mag : float
        Magnitude of the applied force in Newtons
    tau : float
        Time step duration in seconds
    """
    def __init__(self, param_dict, **kwargs):
        """
        Initialize the modified CartPole environment with custom parameters.
        
        Parameters
        ----------
        param_dict : Dict[str, float]
            Dictionary containing the physical parameters of the environment.
            Expected keys: 'masscart', 'masspole', 'length', 'force_mag', 'tau'
        **kwargs : dict
            Additional arguments passed to the base CartPoleEnv constructor
        """
        super().__init__(**kwargs)
        self.masscart = param_dict.get("masscart")
        self.masspole = param_dict.get("masspole")
        self.length = param_dict.get("length")
        self.force_mag = param_dict.get("force_mag")
        self.tau = param_dict.get("tau")

class CartPoleTaskDistribution:
    """
    A task distribution for generating diverse CartPole environments.
    
    This class creates and manages a collection of CartPole tasks with varying
    physical parameters, suitable for meta-learning algorithms that require
    multiple related but distinct tasks.
    
    Attributes
    ----------
    num_tasks : int
        Total number of tasks in the distribution
    task_params : List[Dict[str, float]]
        List of parameter dictionaries for each task
    """
    def __init__(self, num_tasks: int = 100):
        """
        Initialize the task distribution.
        
        Parameters
        ----------
        num_tasks : int, optional
            Total number of tasks to generate, by default 100
        """
        self.num_tasks = num_tasks
        self.task_params = self._generate_task_parameters()
    
    def _generate_task_parameters(self) -> List[Dict[str, float]]:
        """
        Generate random parameters for each task in the distribution.
        
        Returns
        -------
        List[Dict[str, float]]
            List of dictionaries, each containing parameters for one task
        """
        tasks = []
        for _ in range(self.num_tasks):
            task = {
                'masscart': np.random.uniform(0.5, 2.0),    
                'masspole': np.random.uniform(0.05, 0.3),   
                'length': np.random.uniform(0.3, 1.0),      
                'force_mag': np.random.uniform(5.0, 15.0),  
                'tau': np.random.uniform(0.01, 0.05)        
            }
            tasks.append(task)
        return tasks
    
    def sample_tasks(self, num_tasks: int, mode: str = 'train') -> List[Dict[str, float]]:
        """
        Sample a batch of tasks from the distribution.
        
        Parameters
        ----------
        num_tasks : int
            Number of tasks to sample
        mode : str, optional
            Sampling mode: 'train' for training tasks, 'test' for test tasks, by default 'train'
            
        Returns
        -------
        List[Dict[str, float]]
            List of sampled task parameter dictionaries
        """
        if mode == 'train':
            indices = np.random.choice(len(self.task_params[:int(0.8*self.num_tasks)]), 
                                     num_tasks, replace=False)
        else:
            indices = np.random.choice(len(self.task_params[int(0.8*self.num_tasks):]), 
                                     num_tasks, replace=False)
            indices += int(0.8*self.num_tasks)
        
        return [self.task_params[i] for i in indices] 
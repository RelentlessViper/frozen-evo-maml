import torch
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import pandas as pd
from .functional_policy import MAMLPolicyFunctional
from .modified_cartpole import ModifiedCartPoleEnv

class MAML:
    """
    Model-Agnostic Meta-Learning (MAML) implementation for reinforcement learning.
    
    MAML learns a policy initialization that can quickly adapt to new tasks with
    few gradient steps. The algorithm consists of inner loop (task-specific adaptation)
    and outer loop (meta-parameter update) phases.
    
    Attributes
    ----------
    task_dist : TaskDistribution
        Distribution of tasks for meta-training
    inner_lr : float
        Learning rate for inner loop adaptation
    meta_lr : float
        Learning rate for outer loop meta-update
    num_inner_steps : int
        Number of gradient steps in the inner loop
    eval_fr : int
        Evaluation frequency during training
    policy : MAMLPolicyFunctional
        The neural network policy being meta-trained
    meta_optimizer : torch.optim.Optimizer
        Optimizer for the meta-parameters
    """
    def __init__(self, task_distribution, inner_lr=0.01, meta_lr=0.001, num_inner_steps=1, eval_fr=5):
        """
        Initialize the MAML algorithm.
        
        Parameters
        ----------
        task_distribution : TaskDistribution
            Distribution object that can sample training and test tasks
        inner_lr : float, optional
            Learning rate for inner loop adaptation, by default 0.01
        meta_lr : float, optional
            Learning rate for meta-optimizer, by default 0.001
        num_inner_steps : int, optional
            Number of gradient steps in inner loop, by default 1
        eval_fr : int, optional
            Evaluation frequency (iterations), by default 5
        """
        self.task_dist = task_distribution
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.eval_fr = eval_fr
        self.policy = MAMLPolicyFunctional()
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=meta_lr)

    def compute_loss(self, env, params=None, num_rollouts=10, max_steps=500):
        """
        Compute policy gradient loss for a given environment and parameters.
        
        Parameters
        ----------
        env : gym.Env
            Environment to evaluate the policy on
        params : dict, optional
            Policy parameters to use for evaluation. If None, uses current policy parameters
        num_rollouts : int, optional
            Number of trajectory rollouts to average over, by default 10
        max_steps : int, optional
            Maximum steps per rollout, by default 500
            
        Returns
        -------
        tuple
            (average_loss, average_rewards) across all rollouts
        """
        if params is None:
            params = self.policy.state_dict()
        loss = 0
        for _ in range(num_rollouts):
            state, _ = env.reset()
            action_log_probs = []
            rewards = []
            total_rewards = 0
            for step in range(max_steps):
                state_tensor = torch.tensor(state).float()
                action_logits = self.policy(state_tensor, params=params) 
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action)
                action_log_probs.append(action_log_prob)
                
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                rewards.append(reward)
                state = next_state
                
                if terminated or truncated:
                    break

            action_log_probs_tensor = torch.stack(action_log_probs)
            total_rewards = total_rewards + np.sum(rewards)
            rewards_tensor = torch.tensor(rewards).to(action_log_probs_tensor.device)
            current_loss = - (action_log_probs_tensor * rewards_tensor).sum()
            loss = loss + current_loss
        
        avg_loss = loss / num_rollouts
        avg_rewards = total_rewards / num_rollouts
        return (avg_loss, avg_rewards)
    
    def inner_update(self, task_params, num_rollouts=5):
        """
        Perform inner loop adaptation on a specific task.
        
        Parameters
        ----------
        task_params : dict
            Task-specific parameters for environment creation
        num_rollouts : int, optional
            Number of rollouts per inner step, by default 5
            
        Returns
        -------
        tuple
            (adapted_params_dict, final_loss, final_rewards) after inner loop adaptation
        """
        env = ModifiedCartPoleEnv(task_params)
        adapted_params = dict(self.policy.named_parameters())
    
        for inner_step in range(self.num_inner_steps):
            loss, avg_rewards = self.compute_loss(env, params=adapted_params, num_rollouts=num_rollouts)
            gradients = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
        
            new_adapted_params = {}
            for (name, param), grad in zip(adapted_params.items(), gradients):
                new_adapted_params[name] = param - self.inner_lr * grad
            adapted_params = new_adapted_params
        
        return (adapted_params, loss, avg_rewards)
    
    def meta_update(self, num_tasks=5, num_rollouts=5):
        """
        Perform one meta-update step using multiple tasks.
        
        Parameters
        ----------
        num_tasks : int, optional
            Number of tasks to sample for meta-update, by default 5
        num_rollouts : int, optional
            Number of rollouts per task, by default 5
            
        Returns
        -------
        tuple
            (average_meta_loss, average_meta_rewards) across all tasks
        """
        self.meta_optimizer.zero_grad()
        tasks = self.task_dist.sample_tasks(num_tasks, mode='train')
        total_meta_loss = 0
        total_meta_rewards = 0
        
        for task_params in tasks:
        
            adapted_params_dict, _, _ = self.inner_update(task_params, num_rollouts)
            
            env = ModifiedCartPoleEnv(task_params)
            query_loss, meta_rewards = self.compute_loss(env, params=adapted_params_dict, num_rollouts=num_rollouts)
            total_meta_loss = total_meta_loss + query_loss
            total_meta_rewards = total_meta_rewards + meta_rewards
    
        avg_meta_loss = total_meta_loss / num_tasks
        avg_meta_rewards = total_meta_rewards / num_tasks
        avg_meta_loss.backward() 
        self.meta_optimizer.step()
        
        return (avg_meta_loss.item(), avg_meta_rewards)

    def train(self, num_iterations=100, meta_batch_size=5):
        """
        Main training loop for MAML.
        
        Parameters
        ----------
        num_iterations : int, optional
            Total number of meta-training iterations, by default 100
        meta_batch_size : int, optional
            Number of tasks per meta-update, by default 5
            
        Returns
        -------
        pandas.DataFrame
            Training results with metrics for each evaluation point
        """
        print("Starting MAML training...")

        results = pd.DataFrame(
            data={
                "episode": [],
                "n-way/m-shot": [],
                "meta_loss": [],
                "meta_rewards": [],
                "test_loss": [],
                "test_rewards": [],
            }
        )
        train_setting = f"{meta_batch_size}/{self.num_inner_steps}"
        for iteration in range(num_iterations):
            meta_loss, meta_rewards = self.meta_update(num_tasks=meta_batch_size)
            
            if (iteration + 1) % self.eval_fr == 0:
            
                test_loss, test_rewards = self.evaluate(num_test_tasks=3)
                results.loc[results.shape[0]] = [iteration + 1, train_setting, meta_loss, meta_rewards, test_loss, test_rewards]
                print(f"Iteration {iteration + 1}: Meta Loss = {meta_loss:.4f}, Test Loss = {test_loss:.4f}")
                print(f"Meta avg. rewards = {meta_rewards:.4f}, Test avg. rewards = {test_rewards:.4f}")
                print("-" * 10)
    
        print("Training completed!")
        return results
    
    def evaluate(self, num_test_tasks=5):
        """
        Evaluate the meta-trained policy on test tasks.
        
        Parameters
        ----------
        num_test_tasks : int, optional
            Number of test tasks to evaluate on, by default 5
            
        Returns
        -------
        tuple
            (average_test_loss, average_test_rewards) across all test tasks
        """
        test_tasks = self.task_dist.sample_tasks(num_test_tasks, mode='test')
        total_test_loss = 0
        total_test_rewards = 0
        for task_params in test_tasks:
        
            adapted_params_dict, _, _ = self.inner_update(task_params)
            
            env = ModifiedCartPoleEnv(task_params)
            with torch.no_grad():
                test_loss, test_rewards = self.compute_loss(env, params=adapted_params_dict, num_rollouts=10)
            total_test_loss = total_test_loss + test_loss.item()
            total_test_rewards = total_test_rewards + test_rewards

        avg_test_loss = total_test_loss / num_test_tasks
        avg_test_rewards = total_test_rewards / num_test_tasks
        return (avg_test_loss, avg_test_rewards)
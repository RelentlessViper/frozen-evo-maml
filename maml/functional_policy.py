import torch
import torch.nn as nn

class MAMLPolicyFunctional(nn.Module):
    """
    A functional neural network policy that can accept external parameters for forward computation.
    
    This module is designed for meta-learning scenarios like MAML (Model-Agnostic Meta-Learning),
    where the same network architecture needs to be evaluated with different parameter sets
    during the inner loop adaptation process.
    
    Attributes
    ----------
    l1 : nn.Linear
        First linear layer mapping from state dimension to hidden dimension
    l2 : nn.Linear
        Second linear layer mapping from hidden dimension to hidden dimension  
    l3 : nn.Linear
        Third linear layer mapping from hidden dimension to action dimension
    """
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        """
        Initialize the functional policy network.
        
        Parameters
        ----------
        state_dim : int, optional
            Dimension of input state space, by default 4
        action_dim : int, optional  
            Dimension of output action space, by default 2
        hidden_dim : int, optional
            Dimension of hidden layers, by default 128
        """
        super(MAMLPolicyFunctional, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x, params=None):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, state_dim)
        params : dict, optional
            Dictionary containing network parameters. If None, uses the module's 
            current state_dict. Expected keys: 'l1.weight', 'l1.bias', 'l2.weight', 
            'l2.bias', 'l3.weight', 'l3.bias'
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, action_dim)
        """
        if params is None:
            params = self.state_dict()
            
        x = torch.relu(torch.nn.functional.linear(x, params['l1.weight'], params['l1.bias']))
        x = torch.relu(torch.nn.functional.linear(x, params['l2.weight'], params['l2.bias']))
        x = torch.nn.functional.linear(x, params['l3.weight'], params['l3.bias'])
        return x
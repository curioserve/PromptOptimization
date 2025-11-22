"""
INSTINCT: INSTruction optimization usIng Neural bandits Coupled with Transformers

A simplified implementation that integrates with the existing prompt optimization framework.
This version uses neural bandits (NeuralTSDiag) instead of Gaussian Processes for optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.quasirandom import SobolEngine

# Try to import backpack, but make it optional
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
    BACKPACK_AVAILABLE = True
except ImportError:
    BACKPACK_AVAILABLE = False
    print("Warning: backpack-for-pytorch not available. Using simplified gradient computation.")


class Network(nn.Module):
    """Neural network for neural bandit surrogate model."""
    def __init__(self, input_dim, hidden_size=100, depth=1, init_params=None):
        super(Network, self).__init__()
        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))
        
        if init_params is None:
            for i in range(len(self.layer_list)):
                torch.nn.init.normal_(self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]
    
    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y


class NeuralTSDiag:
    """
    Neural Thompson Sampling with diagonal approximation.
    This is the core neural bandit algorithm from INSTINCT.
    """
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, 
                 diagonalize=True, device=None):
        self.diagonalize = diagonalize
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tkwargs = {"device": self.device, "dtype": torch.double}
        
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
        
        # Use extend if backpack is available, otherwise use regular network
        if BACKPACK_AVAILABLE:
            self.func = extend(Network(input_dim).to(**self.tkwargs))
        else:
            self.func = Network(input_dim).to(**self.tkwargs)
        
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.context_list = init_x.to(dtype=torch.float32, device=self.device)
        else:
            self.context_list = None
        if init_y is not None:
            self.reward = init_y.to(dtype=torch.float32, device=self.device)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

        if self.diagonalize:
            self.U = lamdba * torch.ones((self.total_param,), device=self.device)
        else:
            self.U = lamdba * torch.diag(torch.ones((self.total_param,), device=self.device))
        
        self.nu = nu
        self.style = style
        self.loss_func = nn.MSELoss()
        self.mean = None
        self.std = None

    def select(self, context, batch_size=300):     
        """Select next arm using UCB or Thompson Sampling."""
        if self.mean is not None:
            context_ = (context - self.mean) / self.std   
        else:
            context_ = context
        
        context_size = context_.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        g_list = []
        mu = []
        
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context_[(i*batch_size):]
            else:
                context_batch = context_[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            sum_mu = torch.sum(mu_)
            
            if BACKPACK_AVAILABLE:
                with backpack(BatchGrad()):
                    sum_mu.backward()                
                g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() 
                                   for p in self.func.parameters()], dim=1)
            else:
                # Simplified gradient computation without backpack
                g_list_ = []
                for batch_idx in range(context_batch.shape[0]):
                    self.func.zero_grad()
                    mu_single = self.func(context_batch[batch_idx:batch_idx+1])
                    mu_single.backward()
                    grad_vec = torch.cat([p.grad.flatten().detach() 
                                        for p in self.func.parameters()])
                    g_list_.append(grad_vec.unsqueeze(0))
                g_list_ = torch.cat(g_list_, dim=0)
            
            g_list.append(g_list_.cpu())
            mu.append(mu_.cpu())
        
        g_list = torch.vstack(g_list)
        mu = torch.vstack(mu)

        if self.diagonalize:
            sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U.cpu(), dim=1))
        else:
            tmp = torch.matmul(g_list, torch.inverse(self.U.cpu()))
            sigma = torch.sqrt(self.nu * self.lamdba * torch.matmul(tmp, torch.transpose(g_list, 0, 1)))
            sigma = torch.diagonal(sigma, 0)

        if self.style == 'ts':
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif self.style == 'ucb':
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)

        if self.diagonalize:
            self.U += g_list[arm].to(self.device) * g_list[arm].to(self.device)
        else:
            self.U += torch.outer(g_list[arm].to(self.device), g_list[arm].to(self.device))

        return arm, g_list[arm].norm().item()

    def train(self, context, reward, local_training_iter=30):
        """Train the neural network surrogate on new data."""
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        
        if context is not None:
            if self.context_list is None:
                self.context_list = torch.from_numpy(context.reshape(1, -1)).to(**self.tkwargs)
                self.reward = torch.tensor([reward]).to(**self.tkwargs)
            else:
                self.context_list = torch.cat((self.context_list, 
                                             context.reshape(1, -1).to(**self.tkwargs)))
                self.reward = torch.cat((self.reward, 
                                      torch.tensor([reward]).reshape(1,-1).to(**self.tkwargs)))

        self.len = self.context_list.shape[0]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3, 
                                    weight_decay=self.lamdba / self.len)

        self.std = self.context_list.std(dim=0) + 1e-30
        self.mean = self.context_list.mean(dim=0)
        standardized_context = (self.context_list - self.mean) / self.std 
        standardized_reward = self.reward.reshape(-1)
        
        for _ in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(standardized_context).view(-1)
            loss = self.loss_func(pred, standardized_reward)
            loss.backward()
            optimizer.step()
        
        return self.func.state_dict()


class INSTINCT:
    """
    INSTINCT: INSTruction optimization usIng Neural bandits Coupled with Transformers
    
    This class performs instruction optimization using neural bandits instead of 
    Gaussian Processes. It works in a low-dimensional latent space and uses
    embeddings to convert latent vectors to tokens.
    """
    
    def __init__(self, scoring_func, embedding_matrix, token_list,
                 generator_func=None,
                 n_tokens=5, latent_dim=50,
                 random_proj_matrix=None,
                 lambda_reg=0.1, nu=0.1,
                 local_training_iter=30,
                 n_domain=1000, n_eval=100,
                 device=None):
        """
        Initialize the INSTINCT optimizer.
        
        Parameters:
        - scoring_func: function(instruction_text) -> float  
            Function that evaluates a candidate instruction on the target task.
        - embedding_matrix: np.ndarray of shape (V, D)  
            Embedding vectors for a vocabulary of tokens (e.g., GPT-2 embeddings).
        - token_list: list or array of length V  
            List of token strings corresponding to rows of embedding_matrix.
        - generator_func: function(token_sequence_str) -> str, optional  
            Function that generates an instruction text from seed tokens.
        - n_tokens: int, default 5  
            Number of tokens in the prompt.
        - latent_dim: int, default 50  
            Dimensionality of the latent representation.
        - random_proj_matrix: np.ndarray of shape (D, latent_dim), optional  
            Fixed random projection matrix. If None, one is sampled.
        - lambda_reg: float, default 0.1  
            Regularization parameter for neural bandit.
        - nu: float, default 0.1  
            Exploration parameter for neural bandit.
        - local_training_iter: int, default 30  
            Number of training iterations for neural network.
        - n_domain: int, default 1000  
            Number of candidate points in the domain.
        - n_eval: int, default 100  
            Number of points to evaluate at each iteration.
        - device: torch.device, optional  
            Device to use for computation.
        """
        self.scoring_func = scoring_func
        self.embedding_matrix = embedding_matrix
        self.token_list = token_list
        self.generator_func = generator_func
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.local_training_iter = local_training_iter
        self.n_domain = n_domain
        self.n_eval = n_eval
        
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create random projection matrix
        if random_proj_matrix is None:
            np.random.seed(42)
            self.proj_matrix = np.random.randn(embedding_matrix.shape[1], latent_dim)
        else:
            self.proj_matrix = random_proj_matrix
        
        # Precompute embedding norms for efficient distance calculations
        self.embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
        
        # Storage for optimization
        self.X_train = []
        self.y_train = []
        self.best_instruction = None
        self.best_score = -np.inf
        
        # Initialize domain using Sobol sequence
        self.sobol_engine = SobolEngine(dimension=latent_dim, scramble=True, seed=0)
        self.all_X_latent = self.sobol_engine.draw(n_domain)
        
        # Project to embedding space
        self.all_X_embed = torch.tensor(
            self.all_X_latent @ self.proj_matrix.T, 
            dtype=torch.float32
        )
        
        # Neural bandit will be initialized after first evaluations
        self.neural_bandit = None
        self.context_dim = None

    def latent_to_tokens(self, latent_vector):
        """
        Convert a latent vector to discrete tokens.
        
        Args:
            latent_vector: np.ndarray of shape (latent_dim,)
        
        Returns:
            List of token strings
        """
        # Project to embedding space
        projected = latent_vector @ self.proj_matrix.T  # shape: (embedding_dim,)
        
        # Reshape to get n_tokens embeddings
        # Each token gets embedding_dim / n_tokens dimensions, then we find nearest neighbor
        embedding_dim = self.embedding_matrix.shape[1]
        tokens_per_dim = embedding_dim // self.n_tokens
        
        tokens = []
        for i in range(self.n_tokens):
            # Get the portion of the projected vector for this token
            start_idx = i * tokens_per_dim
            end_idx = (i + 1) * tokens_per_dim if i < self.n_tokens - 1 else embedding_dim
            
            # Extract token embedding (use full embedding dimension by repeating/padding)
            if end_idx <= projected.shape[0]:
                token_embedding = projected[start_idx:end_idx]
                # Pad to full embedding dimension if needed
                if len(token_embedding) < embedding_dim:
                    # Repeat the pattern to fill the dimension
                    repeat_factor = (embedding_dim + len(token_embedding) - 1) // len(token_embedding)
                    token_embedding = np.tile(token_embedding, repeat_factor)[:embedding_dim]
            else:
                # If we run out of dimensions, use the full projected vector
                token_embedding = projected[:embedding_dim] if len(projected) >= embedding_dim else np.pad(
                    projected, (0, embedding_dim - len(projected)), mode='constant'
                )
            
            # Compute distances to all embeddings
            distances = np.linalg.norm(self.embedding_matrix - token_embedding, axis=1)
            nearest_idx = np.argmin(distances)
            tokens.append(self.token_list[nearest_idx])
        
        return tokens

    def optimize(self, iterations=10, init_points=5, verbose=True):
        """
        Run INSTINCT optimization.
        
        Args:
            iterations: int, number of optimization iterations
            init_points: int, number of initial random evaluations
            verbose: bool, whether to print progress
        
        Returns:
            (best_instruction, best_score) tuple
        """
        if verbose:
            print(f"Starting INSTINCT optimization...")
            print(f"  Latent dimension: {self.latent_dim}")
            print(f"  Domain size: {self.n_domain}")
            print(f"  Initial points: {init_points}")
            print(f"  Iterations: {iterations}")
        
        # Phase 1: Initial random exploration
        init_idxs = np.random.choice(self.n_domain, init_points, replace=False)
        X_init = self.all_X_embed[init_idxs]
        
        if verbose:
            print(f"\nPhase 1: Evaluating {init_points} initial points...")
        
        Y_init = []
        for i, x_embed in enumerate(X_init):
            # Convert embedding to latent (inverse projection)
            x_latent = np.linalg.lstsq(self.proj_matrix.T, x_embed.numpy(), rcond=None)[0]
            
            # Convert to tokens
            tokens = self.latent_to_tokens(x_latent)
            token_str = ' '.join(tokens)
            
            # Generate instruction
            if self.generator_func:
                instruction = self.generator_func(token_str)
            else:
                instruction = token_str
            
            # Evaluate
            score = self.scoring_func(instruction)
            Y_init.append(score)
            
            if verbose:
                print(f"  Init {i+1}/{init_points}: Score = {score:.4f}")
                print(f"    Instruction: {instruction[:80]}...")
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_instruction = instruction
        
        # Use embeddings directly as context (simplified approach)
        # In full INSTINCT, we would use transformer hidden states
        self.context_dim = X_init.shape[1]
        
        # Initialize neural bandit
        X_train_context = X_init.to(self.device)
        Y_train_tensor = torch.tensor(Y_init, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        self.neural_bandit = NeuralTSDiag(
            input_dim=self.context_dim,
            lamdba=self.lambda_reg,
            nu=self.nu,
            style='ucb',
            init_x=X_train_context,
            init_y=Y_train_tensor,
            diagonalize=True,
            device=self.device
        )
        
        # Phase 2: Neural bandit optimization
        if verbose:
            print(f"\nPhase 2: Neural bandit optimization ({iterations} iterations)...")
        
        all_X_context = self.all_X_embed.to(self.device)
        
        for t in range(iterations):
            if verbose:
                print(f"\nIteration {t+1}/{iterations}")
            
            # Select next point using neural bandit
            if self.n_domain != self.n_eval:
                selected_idx = np.random.choice(self.n_domain, self.n_eval, replace=False)
                context_subset = all_X_context[selected_idx]
                arm_idx, grad_norm = self.neural_bandit.select(context_subset)
                actual_idx = selected_idx[arm_idx.item()]
            else:
                arm_idx, grad_norm = self.neural_bandit.select(all_X_context)
                actual_idx = arm_idx.item()
            
            # Get the selected embedding
            x_embed = self.all_X_embed[actual_idx]
            
            # Convert to latent
            x_latent = np.linalg.lstsq(self.proj_matrix.T, x_embed.numpy(), rcond=None)[0]
            
            # Convert to tokens and generate instruction
            tokens = self.latent_to_tokens(x_latent)
            token_str = ' '.join(tokens)
            
            if self.generator_func:
                instruction = self.generator_func(token_str)
            else:
                instruction = token_str
            
            # Evaluate
            score = self.scoring_func(instruction)
            
            if verbose:
                print(f"  Selected point {actual_idx}: Score = {score:.4f}")
                print(f"    Instruction: {instruction[:80]}...")
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_instruction = instruction
            
            # Train neural bandit on new observation
            x_context = x_embed.unsqueeze(0).to(self.device)
            self.neural_bandit.train(x_context, score, self.local_training_iter)
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best instruction: {self.best_instruction}")
            print(f"Best score: {self.best_score:.4f}")
        
        return self.best_instruction, self.best_score


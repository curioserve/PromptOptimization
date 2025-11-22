import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

class BOInG:
    """
    Bayesian Optimization for Instruction Generation (BOInG)
    
    This class performs BO in a low-dimensional latent space to generate an optimal instruction for a task.
    It uses a random projection into a token embedding space and recovers tokens via nearest neighbors.
    """
    def __init__(self, scoring_func, embedding_matrix, token_list,
                 generator_func=None,
                 n_tokens=5, latent_dim=50,
                 random_proj_matrix=None,
                 lambda_penalty=10.0, beta=2.0,
                 kernel=None, noise=1e-6):
        """
        Initialize the BOInG optimizer.
        
        Parameters:
        - scoring_func: function(instruction_text) -> float  
            Function that evaluates a candidate instruction on the target task, returning a scalar score 
            (higher is better). This encapsulates running the solver model on the task with the given instruction.
        - embedding_matrix: np.ndarray of shape (V, D)  
            Embedding vectors for a vocabulary of tokens (e.g., GPT-2 embeddings). V = vocab size, D = embedding dim.
        - token_list: list or array of length V  
            List of token strings corresponding to rows of embedding_matrix.
        - generator_func: function(token_sequence_str) -> str, optional  
            Function that takes a sequence of seed tokens (concatenated into a string) and generates an instruction text.
            This represents the instruction generator LLM. If None, the token sequence itself is used as the instruction.
        - n_tokens: int, default 5  
            Number of tokens in the prompt (the length of the seed prompt to optimize).
        - latent_dim: int, default 50  
            Dimensionality of the latent representation for each token. (This should be much smaller than D for efficiency.)
        - random_proj_matrix: np.ndarray of shape (D, latent_dim), optional  
            A fixed random projection matrix Φ mapping latent vectors to D-dim embedding space. If None, one is sampled.
        - lambda_penalty: float, default 10.0  
            The weight λ for the distance penalty in the acquisition function (controls how strongly we prefer prompts near known tokens).
        - beta: float, default 2.0  
            The exploration parameter for UCB acquisition. Higher values favor exploration (uncertainty).
        - kernel: sklearn GP kernel, optional  
            Kernel for the Gaussian Process surrogate. If None, uses Matern 5/2 + noise by default.
        - noise: float, default 1e-6  
            Noise level (added to GP diagonal) for numerical stability.
        """
        self.scoring_func = scoring_func
        self.generator_func = generator_func or (lambda token_str: token_str)
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        # Store reference embedding table and vocabulary
        self.embedding_matrix = np.array(embedding_matrix)
        self.token_list = list(token_list)
        self.embed_dim = self.embedding_matrix.shape[1]
        assert self.embedding_matrix.shape[0] == len(self.token_list), \
            "Embedding matrix row count must match token_list length."
        # Initialize or set the random projection matrix Φ
        if random_proj_matrix is not None:
            self.random_proj = np.array(random_proj_matrix)
            assert self.random_proj.shape == (self.embed_dim, self.latent_dim), \
                f"random_proj_matrix must have shape ({self.embed_dim}, {self.latent_dim})."
        else:
            # Sample a random projection matrix (Uniform[-1,1] by default):contentReference[oaicite:22]{index=22}
            self.random_proj = np.random.uniform(-1, 1, size=(self.embed_dim, self.latent_dim))
        # Precompute norms of embedding vectors for efficiency in distance calc
        self.embed_norms = np.sum(self.embedding_matrix**2, axis=1)
        # Set up GP surrogate model
        if kernel is None:
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=noise)  # Matern kernel with noise
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, normalize_y=True)
        self.X_train = [] 
        self.y_train = []  
        self.best_score = -float('inf')
        self.best_latent = None
        self.best_instruction = None
        self.lambda_penalty = lambda_penalty
        self.beta = beta

    def latent_to_tokens(self, latent_prompt):
        """
        Convert a latent prompt to a sequence of tokens via random projection + nearest neighbors.
        
        latent_prompt: np.ndarray shape (n_tokens, latent_dim) or flattened (n_tokens*latent_dim,)
        Returns: list of token strings (length = n_tokens) forming the hard prompt.
        """
        z = np.array(latent_prompt).reshape(self.n_tokens, self.latent_dim)
        tokens = []
        for i in range(self.n_tokens):
            # Project the i-th latent token to embedding space
            embed_vec = self.random_proj.dot(z[i])  # shape (embed_dim,)
            # Find nearest neighbor token in embedding space
            # Compute squared distances: ||e - embed_vec||^2 = ||e||^2 + ||embed_vec||^2 - 2*e·embed_vec
            dists = self.embed_norms + np.sum(embed_vec**2) - 2 * (self.embedding_matrix.dot(embed_vec))
            nn_index = int(np.argmin(dists))
            tokens.append(self.token_list[nn_index])
        return tokens

    def evaluate_latent_prompt(self, latent_prompt):
        """
        Project a latent prompt to tokens, generate an instruction, and evaluate its score.
        Returns: (score, tokens, instruction)
        """
        # 1. Recover hard prompt tokens from latent prompt
        tokens = self.latent_to_tokens(latent_prompt)
        hard_prompt = " ".join(tokens)
        # 2. Generate instruction text using the instruction generator LLM (or identity if none provided)
        instruction = self.generator_func(hard_prompt)
        # 3. Evaluate the instruction on the task using the solver (scoring function)
        score = self.scoring_func(instruction)
        # 4. Update the dataset for GP
        self.X_train.append(np.array(latent_prompt).reshape(-1))
        self.y_train.append(score)
        # Refit GP model to include the new observation
        self.gp.fit(np.vstack(self.X_train), np.array(self.y_train))
        # 5. Update best-found instruction
        if score > self.best_score:
            self.best_score = score
            self.best_latent = np.array(latent_prompt)
            self.best_instruction = instruction
        return score, tokens, instruction

    def acquisition_value(self, latent_prompt_vec):
        """
        Compute the penalized UCB acquisition value for a given latent prompt (flattened vector).
        acq(p) = μ(p) + β * σ(p) - λ * g(p), where g(p) is the average nearest-token distance penalty.
        """
        mu, std = self.gp.predict(latent_prompt_vec.reshape(1, -1), return_std=True)
        mu, std = float(mu), float(std)
        z = latent_prompt_vec.reshape(self.n_tokens, self.latent_dim)
        total_dist = 0.0
        for i in range(self.n_tokens):
            embed_vec = self.random_proj.dot(z[i])
            dists = self.embed_norms + np.sum(embed_vec**2) - 2 * (self.embedding_matrix.dot(embed_vec))
            nn_idx = int(np.argmin(dists))
            total_dist += np.sqrt(dists[nn_idx])
        avg_dist = total_dist / self.n_tokens
        ucb = mu + self.beta * std
        return ucb - self.lambda_penalty * avg_dist

    def propose_next_latent(self, n_candidates=100, bounds=(-1, 1)):
        """
        Suggest the next latent prompt to evaluate by approximately maximizing the acquisition function.
        This uses random sampling (with potential refinements) to find a high-acquisition point.
        
        n_candidates: int, number of random samples to draw in latent space for evaluation.
        bounds: tuple (min, max) for uniform sampling of latent values.
        Returns: np.ndarray of shape (n_tokens, latent_dim) for the next latent prompt.
        """
        best_val = -float('inf')
        best_vec = None
        low, high = bounds
        for _ in range(n_candidates):
            cand = np.random.uniform(low, high, size=(self.n_tokens * self.latent_dim,))
            val = self.acquisition_value(cand)
            if val > best_val:
                best_val = val
                best_vec = cand
        return best_vec.reshape(self.n_tokens, self.latent_dim)

    def optimize(self, iterations=10, init_points=5, verbose=True):
        """
        Run the BOInG optimization loop.
        
        iterations: int, number of BO iterations to perform (after initial random evaluations).
        init_points: int, number of initial random latent prompts to evaluate before starting BO.
        verbose: bool, if True, prints progress of each iteration.
        
        Returns: (best_instruction, best_score)
        """
        for j in range(init_points):
            latent = np.random.uniform(-1, 1, size=(self.n_tokens, self.latent_dim))
            score, tokens, instr = self.evaluate_latent_prompt(latent)
            if verbose:
                print(f"[Initial] Prompt {j+1}: score={score:.4f}, tokens={tokens}, instruction=\"{instr}\"")
        for i in range(1, iterations+1):
            latent = self.propose_next_latent()
            score, tokens, instr = self.evaluate_latent_prompt(latent)
            if verbose:
                print(f"[Iteration {i}] score={score:.4f}, tokens={tokens}, instruction=\"{instr}\"")
        if verbose:
            print(f"Best instruction found: \"{self.best_instruction}\" (score={self.best_score:.4f})")
        return self.best_instruction, self.best_score

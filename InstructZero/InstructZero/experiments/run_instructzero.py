import random
import torch
import numpy as np
import copy
from automatic_prompt_engineer import ape, data
from data.instruction_induction.load_data import load_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from automatic_prompt_engineer import evaluate, config, template, data
import os
import re
from misc import get_test_conf, get_conf

from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from instruction_coupled_kernel import *
import time

from misc import set_all_seed, TASKS, tkwargs, N_INIT, BATCH_SIZE, N_ITERATIONS

from args import parse_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"

    
class LMForwardAPI:
    def __init__(self, model_name=None, eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, random_proj=None, intrinsic_dim=None, n_prompt_tokens=None, few_shot_data=None, 
                 HF_cache_dir=None, args=None):
        p = torch.ones(10)
        
        kwargs={
            'torch_dtype': torch.float16,
            'use_cache': True
            }
        self.ops_model = model_name
        print(f"[LMForwardAPI.__init__] model_name={model_name}, HF_cache_dir={HF_cache_dir}", flush=True)
        # import pdb; pdb.set_trace()
        if self.ops_model in ["vicuna", "wizardlm", 'openchat', 'gpt-oss-20b']:
            print("[LMForwardAPI.__init__] Loading model.from_pretrained...", flush=True)
            _t0 = time.time()
            # Build a max_memory map to utilize all visible GPUs and reduce CPU offload
            max_memory = {}
            cpu_mem = os.getenv('CPU_MAX_MEMORY', '64GiB')
            if torch.cuda.is_available():
                util_frac = float(os.getenv('GPU_UTILIZATION_FRACTION', '0.90'))
                reserve_gb = float(os.getenv('GPU_MEMORY_RESERVE_GB', '0'))
                num_gpus = torch.cuda.device_count()
                print(f"[LMForwardAPI.__init__] torch.cuda.device_count()={num_gpus}", flush=True)
                for i in range(num_gpus):
                    total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    alloc_gb = max(1, int(total_gb * util_frac - reserve_gb))
                    # IMPORTANT: accelerate expects integer GPU ids as keys, not 'cuda:i'
                    max_memory[i] = f"{alloc_gb}GiB"
            max_memory['cpu'] = cpu_mem
            print(f"[LMForwardAPI.__init__] max_memory={max_memory}", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_cache_dir,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory=max_memory,
                **kwargs,
            )
            print(f"[LMForwardAPI.__init__] Model loaded in {time.time()-_t0:.2f}s", flush=True)
            try:
                print(f"[LMForwardAPI.__init__] device_map={getattr(self.model, 'hf_device_map', None)}", flush=True)
            except Exception:
                pass

            print("[LMForwardAPI.__init__] Loading tokenizer.from_pretrained...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                                HF_cache_dir,
                                model_max_length=1024,
                                padding_side="right",
                                use_fast=False,
                            )
            print("[LMForwardAPI.__init__] Tokenizer loaded", flush=True)
        else:
            raise NotImplementedError

        self.init_token = init_prompt[0] + init_qa[0]
        if self.ops_model in ['wizardlm', 'vicuna', 'openchat', 'gpt-oss-20b']:
            print('[LMForwardAPI.__init__] Building input embeddings for init prompt...', flush=True)
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]
            # Stats for scaling soft prompt to embedding manifold
            self.embed_mu = self.embedding.reshape(-1).mean().item()
            self.embed_std = self.embedding.reshape(-1).std().item()
            
        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape), flush=True)
        
        # self.init_prompt = self.init_prompt.reshape(self.n_prompt_tokens * self.hidden_size)
        # Create the template for Vicuna and WizardLM
        self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False)
        if self.ops_model in ['vicuna', 'wizardlm', 'openchat', 'gpt-oss-20b']:
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'alpaca':
            self.system_prompt= "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.role = ["### Instruction:", "### Response:"]
        else:
            raise NotImplementedError
            

        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['wizardlm', 'vicuna', 'openchat', 'gpt-oss-20b']:
                print('Get the embedding firstly to avoid issues', flush=True)
            else:
                raise NotImplementedError
            mu_hat = self.embed_mu
            std_hat = self.embed_std
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(intrinsic_dim) * args.sigma)

            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std), flush=True)
            torch.nn.init.normal_(self.linear.weight, -1, 1)
        elif random_proj == 'uniform':  
            # Start with a very small soft prompt so the base prompt dominates initially
            torch.nn.init.uniform_(self.linear.weight, -1e-3, 1e-3)

        ## eval preparation
        print('[LMForwardAPI.__init__] Updating config...', flush=True)
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")

        # Temporarily remove the API model "LLaMA-33B" and "Flan-T5 13B" 
        # if args.api_model in ['llama', 'flan-t5']:
        #     self.api_model = exec_evaluator(args.api_model, self.conf)
        # else:
        self.api_model = args.api_model
        print(f"[LMForwardAPI.__init__] api_model={self.api_model}", flush=True)

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data
        print(f"[LMForwardAPI.__init__] few_shot_data size={len(self.few_shot_data[0]) if self.few_shot_data is not None else 0}", flush=True)
        
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        print('[LMForwardAPI.__init__] Initialization complete.', flush=True)

    def eval(self, prompt_embedding=None, test_data=None):
        print('[LMForwardAPI.eval] Enter', flush=True)
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        print(f"[LMForwardAPI.eval] prompt_embedding type={type(prompt_embedding)}", flush=True)
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
        
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            # if self.init_prompt is not None:
            #     prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        elif isinstance(prompt_embedding, torch.Tensor): 
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        # create the input text with the system prompt  
        # Encourage a clean single-sentence instruction (plain, non-chat template)
        input_text = (
            "Using the following input/output examples, write one concise English instruction describing the task. "
            "Only output the instruction in a single sentence starting with an imperative verb.\n\n"
            f"{self.init_token}\n\nInstruction:"
        )
        print('[LMForwardAPI.eval] Tokenizing input_text...', flush=True)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
        # Scale soft prompt to the embedding manifold and anchor to init prompt
        prompt_embedding = torch.tanh(prompt_embedding) * self.embed_std + self.embed_mu
        if self.init_prompt is not None:
            prompt_embedding = prompt_embedding + self.init_prompt  # broadcast over prompt tokens
        input_embed = torch.cat((prompt_embedding, input_embed), 1)

        print('[LMForwardAPI.eval] Calling model.generate...', flush=True)
        _tgen = time.time()
        # Build attention mask for inputs_embeds
        attn_mask = torch.ones(input_embed.shape[:2], dtype=torch.long, device=input_embed.device)
        outputs = self.model.generate(
            inputs_embeds=input_embed,
            attention_mask=attn_mask,
            max_new_tokens=128,
            min_new_tokens=16,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=6,
            repetition_penalty=1.1,
        )
        print(f"[LMForwardAPI.eval] model.generate done in {time.time()-_tgen:.2f}s", flush=True)
        # Decode only newly generated tokens (exclude prompt tokens)
        init_len = input_embed.shape[1]
        gen_tokens = outputs[:, init_len:]
        instruction = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        print(f"[LMForwardAPI.eval] Decoded instruction length={len(instruction[0]) if instruction else 0}", flush=True)
        # postprocess instruction: keep a clean single sentence with letters
        try:
            import re
            raw = instruction[0]
            # Remove non-printable chars
            raw = re.sub(r"[^\x20-\x7E]", " ", raw)
            # Take the first line/sentence with alphabets
            candidates = re.split(r"[\n]+|(?<=[\.!?])\s+", raw)
            candidates = [c.strip() for c in candidates if any(ch.isalpha() for ch in c)]
            # Prefer a candidate that doesn't echo the header
            header_phrases = ("below are input/output examples", "using the following input/output examples")
            if candidates:
                chosen = None
                for c in candidates:
                    lc = c.lower()
                    if all(h not in lc for h in header_phrases) and len(c) >= 12:
                        chosen = c
                        break
                cleaned = chosen if chosen is not None else candidates[0]
                # Collapse repeated punctuation
                cleaned = re.sub(r"[\.,;:!?\-]{2,}", lambda m: m.group(0)[0], cleaned)
                # Trim leading/trailing punctuation
                cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned)
                instruction[0] = cleaned
        except Exception as e:
            print(f"[LMForwardAPI.eval] postprocess error: {e}", flush=True)

        # If the instruction is still non-linguistic (e.g., mostly punctuation), optionally fall back to text-only induction
        disable_fb = os.getenv('INDUCTION_DISABLE_FALLBACK', '1') == '1'
        if (not any(ch.isalpha() for ch in instruction[0]) or len(instruction[0]) < 10) and not disable_fb:
            print('[LMForwardAPI.eval] Fallback to text-only induction (no soft prompt)', flush=True)
            plain_text = (
                "Using the following input/output examples, write one concise English instruction describing the task. "
                "Only output the instruction in a single sentence starting with an imperative verb.\n\n"
                f"{self.init_token}\n\nInstruction:"
            )
            enc = self.tokenizer(plain_text, return_tensors="pt")
            input_ids = enc.input_ids.to(self.model.device)
            attn_mask = enc.attention_mask.to(self.model.device) if 'attention_mask' in enc else None
            outputs2 = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=32,
                min_new_tokens=16,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=6,
                repetition_penalty=1.1,
            )
            # Decode only newly generated tokens
            input_len = input_ids.shape[1]
            gen2 = outputs2[:, input_len:]
            inst2 = self.tokenizer.batch_decode(gen2, skip_special_tokens=True)
            # Clean again
            try:
                import re
                raw2 = inst2[0]
                raw2 = re.sub(r"[^\x20-\x7E]", " ", raw2)
                cands = re.split(r"[\n]+|(?<=[\.!?])\s+", raw2)
                cands = [c.strip() for c in cands if any(ch.isalpha() for ch in c)]
                if cands:
                    banned = ("input", "output", "examples", "instruction")
                    chosen2 = None
                    for c in cands:
                        cl = c.lower()
                        if all(b not in cl for b in banned) and len(c) >= 12:
                            chosen2 = c
                            break
                    cleaned2 = chosen2 if chosen2 is not None else cands[0]
                    cleaned2 = re.sub(r"[\.,;:!\?\-]{2,}", lambda m: m.group(0)[0], cleaned2)
                    cleaned2 = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned2)
                    instruction[0] = cleaned2
            except Exception as e:
                print(f"[LMForwardAPI.eval] text-only postprocess error: {e}", flush=True)
        # instruction[0] = 'The instruction was to ' + instruction[0]
        # import pdb; pdb.set_trace()
        # start = instruction[0].find('The instruction was to')
        # end = instruction[0].find('Comment:')
        # if end == -1:
        #     instruction[0] = instruction[0][start:]
        # else:
        #     instruction[0] = instruction[0][start: end]

        # sentences = re.split(r' *[\.\?!][\'"\)\]]* *', instruction[0])
        # search_string = 'The instruction was to'
        # for sentence in sentences:
        #     if sentence.startswith(search_string):
        #         instruction[0] = sentence
        #         break

        # print post-processed instruction
        print('Instruction: {}'.format(instruction), flush=True)
        
        if instruction[0] in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
        else:
            print(f"[LMForwardAPI.eval] Evaluating prompts via api_model={self.api_model}", flush=True)
            if self.api_model in ['chatgpt']:
                _tev = time.time()
                dev_perf, instruction_score = evaluate.evaluate_prompts(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation']['method'], self.conf['evaluation'])
                print(f"[LMForwardAPI.eval] evaluate_prompts done in {time.time()-_tev:.2f}s", flush=True)
                dev_perf = dev_perf.sorted()[1][0]
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            # We will fix the bugs for other api models. Stay tuned!
            # elif api_model in ['llama', 'flan-t5']: 
            #     dev_perf, instruction_score = self.api_model.evaluate(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data,
            #                             self.conf['evaluation']).sorted()[1][0]            
            #     self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            else:
                raise NotImplementedError

        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_instruction = instruction

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********', flush=True)

        return dev_perf, instruction_score

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set
    
def run(args):
    print('[run] Starting run(args)', flush=True)
    task, HF_cache_dir = args.task, args.HF_cache_dir
    # Accept tasks provided as "name.json" by stripping the extension
    if isinstance(task, str) and task.endswith('.json'):
        print(f"[run] Stripping .json from task '{task}'", flush=True)
        task = task[:-5]
    # Update args.task to normalized form for downstream code
    args.task = task
    random_proj, intrinsic_dim, n_prompt_tokens= args.random_proj, args.intrinsic_dim, args.n_prompt_tokens
    print(f"[run] task={task} | HF_cache_dir={HF_cache_dir} | random_proj={random_proj} | intrinsic_dim={intrinsic_dim} | n_prompt_tokens={n_prompt_tokens}", flush=True)

    assert task in TASKS, f"Task not found! Provided: {task}. Available: {TASKS}"

    print('[run] Loading data...', flush=True)
    induce_data, test_data = load_data('induce', task), load_data('eval', task)

    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size), 100)
    print(f"[run] induce_data_size={induce_data_size} | prompt_gen_size={prompt_gen_size}", flush=True)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data, eval_data = data.create_split(
        induce_data, prompt_gen_size)
    print(f"[run] prompt_gen_data sizes: inputs={len(prompt_gen_data[0])}, outputs={len(prompt_gen_data[1])}", flush=True)

    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                           for output in prompt_gen_data[1]]
    # import pdb; pdb.set_trace()
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOUTPUT: [OUTPUT]" # change the evaluation template
    init_prompt = ['\n']
    prompt_gen_template = "[full_DEMO]\n\nThe instruction was to?"
    # prompt_gen_template = "[full_DEMO]\n\nWhat was the instruction for the task?"
    # prompt_gen_template = "[full_DEMO]\n\n Please generate appropriate instructions for the task."

    base_conf = '../configs/instruction_induction.yaml'
    conf = get_conf(task, eval_data)
    print('[run] Config prepared for evaluation', flush=True)

    # make the demo automatically
    subsampled_data = data.subsample_data(prompt_gen_data, conf['generation']['num_demos'])
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]
    
    print('[run] Initializing LMForwardAPI...', flush=True)
    model_forward_api = LMForwardAPI(model_name=args.model_name, eval_data=eval_data, init_prompt=init_prompt, 
                                    init_qa=init_qa, conf=conf, base_conf=base_conf, prompt_gen_data=prompt_gen_data, random_proj=random_proj, 
                                    intrinsic_dim=intrinsic_dim, n_prompt_tokens=n_prompt_tokens, HF_cache_dir=HF_cache_dir, args=args)
    print('[run] LMForwardAPI initialized', flush=True)
        
    # start bayesian opt
    print('[run] Drawing initial Sobol points...', flush=True)
    X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(N_INIT)
    print('[run] Evaluating initial points...', flush=True)
    X_return = [model_forward_api.eval(x) for x in X]
    Y = [X[0] for X in X_return]
    Y_scores = [X[1].squeeze() for X in X_return]
    
    X = X.to(**tkwargs)
    Y = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
    Y_scores = torch.FloatTensor(np.array(Y_scores)).to(**tkwargs)
    print(f"Best initial point: {Y.max().item():.3f}", flush=True)

    # standardization Y (no standardization for X)
    X_train = X
    y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2) + 1e-9)

    # define matern kernel
    matern_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=X_train.shape[-1],
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                )
    matern_kernel_instruction = MaternKernel(
                nu=2.5,
                ard_num_dims=Y_scores.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
    
    covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction, latent_train=X_train.double(), instruction_train=Y_scores))
    gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
    gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    
    for i in range(N_ITERATIONS):
        print(f"[run][iter {i}] X_train shape {X_train.shape}", flush=True)
        print(f"[run][iter {i}] y_train shape {y_train.shape}", flush=True)

        start_time = time.time()

        print(f"[run][iter {i}] Fitting GP MLL...", flush=True)
        fit_gpytorch_mll(gp_mll)#, options = {'maxiter':10})
        print(f"[run][iter {i}] Fitting done in {time.time()-start_time}", flush=True)
        start_time = time.time()
        EI = ExpectedImprovement(gp_model, best_f = y_train.max().item())
        print(f"[run][iter {i}] EI prepared", flush=True)
        
        starting_idxs = torch.argsort(-1*y_train.squeeze())[:BATCH_SIZE]
        starting_points = X_train[starting_idxs]


        best_points = []
        best_vals = []
        for starting_point_for_cma in starting_points:
            if (torch.max(starting_point_for_cma) > 1 or torch.min(starting_point_for_cma) < -1):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI, tkwargs)
            best_points.append(newp)
            best_vals.append(newv)
            
        print(f"[run][iter {i}] best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}", flush=True)
        print(f"[run][iter {i}] Time for CMA-ES {time.time() - start_time}", flush=True)
        for idx in np.argsort(-1*np.array(best_vals)):
            X_next_point =  torch.from_numpy(best_points[idx]).float().unsqueeze(0)
            # Y_next_point = [model_forward_api.eval(X_next_point)]
            
            print(f"[run][iter {i}] Evaluating candidate idx={idx}", flush=True)
            X_next_points_return = [model_forward_api.eval(X_next_point)]
            Y_next_point = [X[0] for X in X_next_points_return]
            Y_scores_next_points = [X[1].squeeze() for X in X_next_points_return]
    
            X_next_point = X_next_point.to(**tkwargs)
            Y_next_point = torch.FloatTensor(Y_next_point).unsqueeze(-1).to(**tkwargs)
            Y_scores_next_points = torch.FloatTensor(np.array(Y_scores_next_points)).to(**tkwargs)

            X = torch.cat([X, X_next_point])
            Y = torch.cat([Y, Y_next_point])
            Y_scores = torch.cat([Y_scores, Y_scores_next_points])

        # standardization Y
        X_train = X.clone()
        y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2) + 1e-9)

        matern_kernel = MaternKernel(
                        nu=2.5,
                        ard_num_dims=X_train.shape[-1],
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                    )
        matern_kernel_instruction = MaternKernel(
                nu=2.5,
                ard_num_dims=Y_scores.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction, latent_train=X_train.double(), instruction_train=Y_scores))
        gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        print(f"[run][iter {i}] Best value found till now: {torch.max(Y)}", flush=True)

    print('Evaluate on test data...', flush=True)
    prompts = model_forward_api.return_best_prompt()
    print("Best instruction is:")
    print(prompts)

    print("The final instruction set is:")
    print(model_forward_api.return_prompts_set())

    # Evaluate on test data
    print('Evaluating on test data...', flush=True)

    test_conf = get_test_conf(task, test_data)
    
    test_res = ape.evaluate_prompts(prompts=prompts,
                                    eval_template=eval_template,
                                    eval_data=test_data,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    conf=test_conf,
                                    base_conf=base_conf)
    test_res = test_res[0]
    test_score = test_res.sorted()[1][0]
    return test_score
    # print(f'Test score on ChatGPT: {test_score}')


if __name__ == '__main__':
    args = parse_args()
    # evaluation budget
    print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations")
    print(set_all_seed(args.seed))
    test_score = run(args=args)
    print("Finished!!!")
    print(f'Test score on ChatGPT: {test_score}')

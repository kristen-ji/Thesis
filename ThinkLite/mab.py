from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration, GenerationConfig
import json
import os
import numpy as np
import math
import torch
import torch.nn as nn
from PIL import Image
import requests
import torch.nn.init as init
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from safetensors.torch import load_file
import os
from datasets import load_dataset
import pandas as pd
import io
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
import torch.nn.functional as F

eval_prompt_template = '''Please help me judge the correctness of the generated answer and the corresponding rationale. 
Question: {}
Ground truth answer: {}
Generated rationale and answer: {}
Your output should only be one sentence: the generated answer is true or false.'''

few_shot_cot_prompt = '''Answer the question **step by step** and provide the final answer at the end, each step should end with **<end>** and put your final answer within $\boxed{}$. Below are two examples:
Question: BoatsRUs built 7 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatsRUs by the end of May of this year?
### Step1: To find the result of the total number of canoes built by BoatsRUs by the end of May, I need to find the number of canoes built in each month from January to May and then add them up. <end>
### Step2: To find the number of canoes built in each month, I need to use the formula for the number of canoes built in a given month, which is the number of canoes built in the previous month times 2. <end>
### Step3: So, the number of canoes built in January is 7, the number of canoes built in February is 7 times 2, which is 14, the number of canoes built in March is 14 times 2, which is 28, the number of canoes built in April is 28 times 2, which is 56, and the number of canoes built in May is 56 times 2, which is 112. <end>
### Step4: Now, I can add up these numbers to get the total number of canoes built by BoatsRUs by the end of May: 7 plus 14 plus 28 plus 56 plus 112, which is 217. <end>
### Final Answer: The answer is: $boxed{217}$.
Question: Find the number of blue circles in the figure.
### Step 1: To find the result of the number of blue circles, I need to interpret the figure. The figure is a Venn diagram with two labeled sets: - One set labeled "blue" contains all the shapes that are blue in color. - The other set labeled "circle" contains all the shapes that are circular in shape. The overlapping region of the Venn diagram contains shapes that are both blue and circular. <end>
### Step 2: The overlapping region contains shapes that meet both criteria: Blue color and Circle shape. From the diagram: - There is **one blue circle** in the overlapping region. <end>
### Final Answer: The answer is: $boxed{1}$.
Remember to answer the question **step by step**! Here is your question:
'''


def read_all_parquet_to_list(directory: str):
    parquet_files = [
        f for f in os.listdir(directory) if f.endswith(".parquet")
    ]

    df_list = []

    for parquet_file in parquet_files:
        file_path = os.path.join(directory, parquet_file)
        df = pd.read_parquet(file_path)
        df_list.append(df)

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
    else:
        return []

    data_list = combined_df.to_dict(orient='records')

    return data_list

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    print(f"DEBUG: Total data length: {len(lst)}")
    print(f"DEBUG: Number of chunks: {n}")
    print(f"DEBUG: Requested chunk index: {k}")
    print(f"DEBUG: Available chunk indices: 0-{len(chunks)-1}")
    print(f"DEBUG: Chunk sizes: {[len(chunk) for chunk in chunks]}")
    
    if k >= len(chunks):
        raise IndexError(f"Chunk index {k} out of range. Available chunks: 0-{len(chunks)-1}")
    
    return chunks[k]

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

class State:

    def __init__(self, image_feat, text_context, solution_steps=None):
        self.image_feat = image_feat
        self.text_context = text_context
        self.solution_steps = solution_steps if solution_steps else []
        self.is_terminal = False

    def copy(self):
        new_state = State(
            image_feat=self.image_feat,
            text_context=self.text_context,
            solution_steps=self.solution_steps.copy()
        )
        new_state.is_terminal = self.is_terminal
        return new_state

    def __repr__(self):
        return f"<State steps={len(self.solution_steps)}, terminal={self.is_terminal}>"


class Action:

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f"<Action: {self.text}>"


class Qwen2_5_VL_Embedder:
    """Qwen2.5-VL vision encoder based embedding for multimodal input"""
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def embed(self, image_feat, text):
        """
        Extract vision and text embeddings from Qwen2.5-VL model
        Returns: combined multimodal embedding
        """
        image = Image.open(io.BytesIO(image_feat))
        
        # Prepare inputs
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
            ],
        }]
        
        text_prompt = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            # Extract hidden states from the model
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get the last hidden state
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
            
            # Pool over sequence dimension (mean pooling)
            pooled_output = hidden_states.mean(dim=1)  # [batch, hidden_dim]
            
            # Normalize
            pooled_output = F.normalize(pooled_output, p=2, dim=-1)
            
        return pooled_output


def estimate_difficulty_from_confidence(model, processor, image_feat, text_context):
    """
    Estimate input difficulty using model prediction confidence.
    
    Uses the top-1 prediction probability as a proxy for difficulty:
    - High confidence (0.999) → Easy problem (low difficulty)
    - Low confidence (0.5) → Hard problem (high difficulty)
    
    Args:
        model: VLM model
        processor: Model processor
        image_feat: Image data (bytes)
        text_context: Text prompt
        
    Returns:
        difficulty_score: Higher score = more difficult (0.0 to 1.0)
        confidence: Top-1 prediction probability (for debugging)
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Prepare inputs
    prompt = text_context
    message = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
        ],
    }]
    
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )[:-32]
    image_inputs = Image.open(io.BytesIO(image_feat))
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Single forward pass to get prediction confidence
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Last token logits
        probs = F.softmax(logits, dim=-1)
    
    # Check for NaN or Inf values
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        return 0.5, 0.5
    
    # Get top-1 probability as confidence measure
    top_prob = probs.max().item()
    
    # Transform confidence to difficulty score
    # Apply extremely aggressive transformation to spread out scores
    # Models have very high confidence (0.98-0.9999), need maximum amplification
    # Use exponential scaling: difficulty = (1 - top_prob)^0.08
    # This maps: 0.9999->0.35, 0.999->0.47, 0.99->0.63, 0.95->0.79, 0.9->0.89
    raw_difficulty = 1.0 - top_prob
    # Add small epsilon to avoid numerical issues
    difficulty_score = min(1.0, max(0.0, (raw_difficulty + 1e-10) ** 0.08))
    
    return difficulty_score, top_prob


class VisionLanguageModel:
    def __init__(self, model, processor, clip_embedder=None):
        self.model = model
        self.processor = processor
        self.clip_embedder = clip_embedder

    def _run_vlm(self, image_feat, text_context, generation_config, history=None):

        prompt = text_context
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }]
        if history:
            message.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "".join(history)}, ],
            })

        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )[:-32]
        image_inputs = Image.open(io.BytesIO(image_feat))
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        question_input_length = inputs['input_ids'].shape[1]

        #generated_ids = self.model.generate(**inputs, generation_config=generation_config, stop_strings=['<end>'],
                                       #max_new_tokens=2048, tokenizer=self.processor.tokenizer)
        generated_ids = self.model.generate(**inputs, generation_config=generation_config, stop_strings=['<end>'],
                                       max_new_tokens=512, tokenizer=self.processor.tokenizer)
        output = self.processor.decode(
            generated_ids[0][question_input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output

    def propose_actions(self, state, generation_config, top_k=3):
        actions = []
        # Use temperature sampling for diversity - single call with higher temperature
        temp_config = GenerationConfig(
            temperature=0.8,  # Higher temperature for diversity
            do_sample=True,
            top_p=0.9,
            max_new_tokens=256
        )
        
        # Single model call with temperature sampling
        llama_output = self._run_vlm(
            image_feat=state.image_feat,
            text_context=state.text_context,
            generation_config=temp_config,
            history=state.solution_steps
        )
        
        # Create multiple actions from single output by splitting on step markers
        action_text = llama_output
        prob = 1.0 / top_k
        
        # Split the output into multiple reasoning steps if possible
        if "### Step" in action_text:
            steps = action_text.split("### Step")[1:]  # Skip the first empty part
            for i, step in enumerate(steps[:top_k]):
                if step.strip():
                    step_text = "### Step" + step.strip()
                    actions.append((Action(step_text), prob))
        else:
            # If no step markers, create variations by truncating
            words = action_text.split()
            chunk_size = len(words) // top_k
            for i in range(top_k):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < top_k - 1 else len(words)
                chunk_text = " ".join(words[start_idx:end_idx])
                if chunk_text.strip():
                    actions.append((Action(chunk_text), prob))
        
        # Ensure we have at least one action
        if not actions:
            actions.append((Action(action_text), 1.0))
            
        return actions[:top_k]

    def transition(self, state, action):
        next_state = state.copy()
        next_state.solution_steps.append(action.text)

        if len(next_state.solution_steps) >= 10 or "Final Answer: " in next_state.solution_steps[-1]:
            next_state.is_terminal = True
        return next_state

    def evaluate_terminal_state(self, state, eval_llm, eval_llm_tokenizer, question, answer):
        if state.is_terminal:
            simulation_response = "".join(state.solution_steps)
            prompt = eval_prompt_template.format(question, answer, simulation_response)

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = eval_llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = eval_llm_tokenizer([text], return_tensors="pt").to(eval_llm.device)

            generated_ids = eval_llm.generate(
                **model_inputs,
                #max_new_tokens=512
                max_new_tokens=128
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = eval_llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if 'true' in response.split('.')[0]:
                return 1.0
            else:
                return 0.0
        return 0.0

class UCB:
    def __init__(self, n_arms, exploration_c=2.0):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0
        self.exploration_c = float(exploration_c)

    def select_arm(self):
        confidence = self.exploration_c * np.sqrt(
            np.log(self.total_counts + 1.0 + 1e-9) / (self.counts + 1e-9)
        )
        ucb_values = self.values + confidence
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1.0) / n) * value + (1.0 / n) * float(reward)

def mab_search(root_state, vlm, eval_llm, eval_llm_tokenizer, question, answer, generation_config,
               n_iterations, top_k=3, exploration_c=2.0, rollout_limit=2):
    """Multi-armed bandit search over top_k proposed actions for a single problem.

    At each iteration, propose top_k actions from the current root state, pick one via UCB1,
    perform a short rollout, evaluate a binary reward, and update the bandit. If reward==1, stop.
    Returns a tuple aligned with mcts_search: (root_node, best_path, solution_state, iterations).
    """
    bandit = UCB(n_arms=top_k, exploration_c=exploration_c)
    best_path = []
    solution = None

    for iter in range(n_iterations):
        actions_probs = vlm.propose_actions(root_state, generation_config, top_k=top_k)
        if not actions_probs:
            break

        chosen_idx = bandit.select_arm()
        chosen_idx = max(0, min(chosen_idx, len(actions_probs) - 1))
        action, _ = actions_probs[chosen_idx]

        temp_state = vlm.transition(root_state, action)

        # Short rollout to try to reach terminal and get a label
        steps = 0
        rollout_state = temp_state
        while not rollout_state.is_terminal and steps < rollout_limit:
            next_actions = vlm.propose_actions(rollout_state, generation_config, top_k=1)
            if not next_actions:
                break
            next_action, _ = next_actions[0]
            rollout_state = vlm.transition(rollout_state, next_action)
            steps += 1

        reward = vlm.evaluate_terminal_state(rollout_state, eval_llm, eval_llm_tokenizer, question, answer)
        bandit.update(chosen_idx, reward)

        if reward == 1.0:
            solution = rollout_state
            best_path = rollout_state.solution_steps
            return None, best_path, solution, iter

    return None, best_path, solution, n_iterations


class DifficultyBasedMAB:
    """Multi-Armed Bandit for selecting which difficulty level to solve next"""
    def __init__(self, difficulty_bins=3, exploration_c=2.0, adaptive_bins=True):
        """
        Args:
            difficulty_bins: Number of difficulty levels (arms)
                            e.g., 3 = [easy, medium, hard]
            exploration_c: UCB exploration constant
            adaptive_bins: If True, compute thresholds based on data percentiles
        """
        self.difficulty_bins = difficulty_bins
        self.bandit = UCB(n_arms=difficulty_bins, exploration_c=exploration_c)
        self.adaptive_bins = adaptive_bins
        self.difficulty_thresholds = self._compute_thresholds()
        
    def _compute_thresholds(self, difficulty_scores=None):
        """Compute difficulty thresholds to divide [0, 1] into bins"""
        if self.adaptive_bins and difficulty_scores is not None:
            # Use percentiles to create balanced bins
            percentiles = np.linspace(0, 100, self.difficulty_bins + 1)
            thresholds = np.percentile(difficulty_scores, percentiles)
            # Ensure first is 0 and last is 1
            thresholds[0] = 0.0
            thresholds[-1] = 1.0
            return thresholds
        else:
            # Uniform binning
            return np.linspace(0, 1, self.difficulty_bins + 1)
    
    def update_thresholds(self, difficulty_scores):
        """Update thresholds based on actual difficulty distribution"""
        if self.adaptive_bins:
            self.difficulty_thresholds = self._compute_thresholds(difficulty_scores)
    
    def assign_to_arm(self, difficulty_score):
        """Assign a difficulty score to an arm (bin)"""
        for i in range(self.difficulty_bins):
            if self.difficulty_thresholds[i] <= difficulty_score < self.difficulty_thresholds[i + 1]:
                return i
        return self.difficulty_bins - 1  # Last bin for score = 1.0
    
    def select_difficulty_arm(self):
        """Select which difficulty level to work on next using UCB"""
        return self.bandit.select_arm()
    
    def update(self, arm_idx, reward):
        """Update bandit statistics after solving a problem"""
        self.bandit.update(arm_idx, reward)
    
    def get_arm_stats(self):
        """Get statistics for each difficulty arm"""
        stats = []
        for i in range(self.difficulty_bins):
            stats.append({
                'arm': i,
                'difficulty_range': f"[{self.difficulty_thresholds[i]:.2f}, {self.difficulty_thresholds[i+1]:.2f})",
                'count': int(self.bandit.counts[i]),
                'avg_reward': float(self.bandit.values[i]),
            })
        return stats

def solve_math_reasoning_vlm(image_data, text_prompt, model, generation_config, processor, eval_llm,
                                eval_llm_tokenizer, question, answer, n_iterations,
                                clip_embedder=None, use_difficulty_adaptive=True,
                                search_method='mab', top_k=3, c_puct=1.0, mab_c=2.0, rollout_limit=5):
    image_feat = image_data
    
    # Step 1: Compute CLIP embeddings if available
    clip_embedding = None
    if clip_embedder is not None:
        clip_embedding = clip_embedder.embed(image_feat, question)
    
    # Step 2: Estimate difficulty from model confidence
    difficulty_score = 0.5  # Default medium difficulty
    
    if use_difficulty_adaptive:
        try:
            difficulty_score, top_prob = estimate_difficulty_from_confidence(
                model=model,
                processor=processor,
                image_feat=image_feat,
                text_context=text_prompt
            )
            
            # Adaptive iteration count based on difficulty
            # Easy problems (score < 0.3): use fewer iterations
            # Medium problems (0.3 <= score < 0.7): use default iterations
            # Hard problems (score >= 0.7): use more iterations
            if difficulty_score < 0.3:
                adapted_iterations = max(3, int(n_iterations * 0.6))
            elif difficulty_score >= 0.7:
                adapted_iterations = int(n_iterations * 1.5)
            else:
                adapted_iterations = n_iterations
        except Exception:
            adapted_iterations = n_iterations
    else:
        adapted_iterations = n_iterations

    init_state = State(
        image_feat=image_feat,
        text_context=text_prompt,
        solution_steps=[]
    )

    vlm = VisionLanguageModel(model, processor, clip_embedder=clip_embedder)

    # Always use MAB search with difficulty-adapted iterations
    root, steps, solution, n_iter = mab_search(
        root_state=init_state,
        vlm=vlm,
        eval_llm=eval_llm,
        eval_llm_tokenizer=eval_llm_tokenizer,
        question=question,
        answer=answer,
        generation_config=generation_config,
        n_iterations=adapted_iterations,
        top_k=top_k,
        exploration_c=mab_c,
        rollout_limit=rollout_limit,
    )
    
    return root, steps, solution, n_iter, difficulty_score, clip_embedding


def main(args):
    # Fix CUDA device issues
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Available CUDA devices: {device_count}")
        if args.gpu_id >= device_count:
            args.gpu_id = 0
        device = f"cuda:{args.gpu_id}"
        torch.cuda.set_device(args.gpu_id)
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    generation_config = GenerationConfig(
        temperature=0.2,  # Lower temperature for faster convergence
        do_sample=True,
        top_p=0.7,  # Further reduced for faster generation
        max_new_tokens=128,  # Reduced token limit for speed
    )

    # Load Qwen2.5-VL model (will be used for both generation and embeddings)
    print(f"Loading Qwen2.5-VL model: {args.model_id}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16, 
        device_map="auto" if torch.cuda.is_available() else None,  # Let transformers handle device mapping
        low_cpu_mem_usage=True, 
        use_cache=True,
        attn_implementation="flash_attention_2"  # Use flash attention for speed
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    print("Qwen2.5-VL model loaded successfully")
    
    # Create embedder using the same model
    vision_embedder = Qwen2_5_VL_Embedder(model, processor) if args.use_vision_embeddings else None
    if vision_embedder:
        print("Vision embedder initialized using Qwen2.5-VL encoder")

    eval_llm = AutoModelForCausalLM.from_pretrained(
        args.eval_model_name,
        torch_dtype=torch.float16,  # Use float16 for consistency
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        use_cache=True,
        attn_implementation="flash_attention_2"
    )
    eval_llm_tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name)
    final_response = []

    #df = pd.read_parquet(args.data_pths, engine='pyarrow')  # Your path of dataset
    # Load dataset from Hugging Face
    ds = load_dataset("russwang/ThinkLite-VL-70k")
    df = ds['train'].to_pandas()  # Convert to pandas DataFrame
    datas = df.to_dict(orient='records')

    data_chunk = get_chunk(datas, args.num_chunks, args.chunk_idx)
    
    # Phase 1: Pre-compute difficulty scores for all samples
    print("Phase 1: Computing difficulty scores for all samples...")
    samples_with_difficulty = []
    
    for data in tqdm(data_chunk, desc="Computing difficulties"):
        try:
            image_data = data['image']
            question = data['problem'].split('<image>')[1]
            text_prompt = few_shot_cot_prompt + '{}'.format(question)
            
            # Compute difficulty score from model confidence
            difficulty_score, top_prob = estimate_difficulty_from_confidence(
                model=model,
                processor=processor,
                image_feat=image_data,
                text_context=text_prompt
            )
            
            # Get vision embedding if enabled
            vision_emb = None
            if vision_embedder:
                vision_emb = vision_embedder.embed(image_data, question)
            
            samples_with_difficulty.append({
                'data': data,
                'difficulty': difficulty_score,
                'vision_emb': vision_emb,
                'image_data': image_data,
                'question': question,
                'answer': data['answer'],
                'text_prompt': text_prompt
            })
            
            if len(samples_with_difficulty) % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception:
            continue
    
    # Phase 2: Group samples by difficulty into arms
    print(f"\nPhase 2: Grouping {len(samples_with_difficulty)} samples by difficulty...")
    difficulty_mab = DifficultyBasedMAB(
        difficulty_bins=args.difficulty_bins,
        exploration_c=args.exploration_c,
        adaptive_bins=True  # Use adaptive binning based on data distribution
    )
    
    # Update thresholds based on actual difficulty distribution
    all_difficulties = [s['difficulty'] for s in samples_with_difficulty]
    difficulty_mab.update_thresholds(all_difficulties)
    
    # Assign samples to arms
    arms = [[] for _ in range(args.difficulty_bins)]
    for sample in samples_with_difficulty:
        arm_idx = difficulty_mab.assign_to_arm(sample['difficulty'])
        arms[arm_idx].append(sample)
    
    print("Difficulty distribution:")
    all_difficulties = [s['difficulty'] for s in samples_with_difficulty]
    print(f"  Overall stats - Min: {min(all_difficulties):.3f}, Max: {max(all_difficulties):.3f}, Mean: {np.mean(all_difficulties):.3f}")
    for i, arm_samples in enumerate(arms):
        thresholds = difficulty_mab.difficulty_thresholds
        if len(arm_samples) > 0:
            arm_diffs = [s['difficulty'] for s in arm_samples]
            avg_diff = np.mean(arm_diffs)
            print(f"  Arm {i} [{thresholds[i]:.2f}, {thresholds[i+1]:.2f}): {len(arm_samples)} samples (avg difficulty: {avg_diff:.3f})")
        else:
            print(f"  Arm {i} [{thresholds[i]:.2f}, {thresholds[i+1]:.2f}): {len(arm_samples)} samples")
    
    # Phase 3: Use MAB to select which difficulty arm to solve
    print(f"\nPhase 3: Solving problems with difficulty-based MAB...")
    solved_indices = [set() for _ in range(args.difficulty_bins)]  # Track solved samples per arm
    
    for iteration in tqdm(range(len(samples_with_difficulty)), desc="MAB-guided solving"):
        # Select which difficulty arm to work on (try multiple times if arm is empty)
        selected_arm = None
        available_indices = []
        
        # Try to find a non-empty arm (up to difficulty_bins attempts)
        for attempt in range(args.difficulty_bins):
            candidate_arm = difficulty_mab.select_difficulty_arm()
            candidate_indices = [i for i in range(len(arms[candidate_arm])) 
                                if i not in solved_indices[candidate_arm]]
            
            if candidate_indices:
                # Found an arm with available samples
                selected_arm = candidate_arm
                available_indices = candidate_indices
                break
            else:
                # This arm is empty, penalize it slightly and try another
                # Give a small negative reward to discourage selecting empty arms
                difficulty_mab.update(candidate_arm, 0.0)
        
        if selected_arm is None or not available_indices:
            # All arms are exhausted
            print(f"\nAll arms exhausted at iteration {iteration + 1}/{len(samples_with_difficulty)}")
            break
        
        # Pick a random sample from available ones in this arm
        sample_idx = random.choice(available_indices)
        sample = arms[selected_arm][sample_idx]
        solved_indices[selected_arm].add(sample_idx)
        
        try:
            # Solve the problem
            root, solution_steps, solution, n_iter, _, _ = solve_math_reasoning_vlm(
                image_data=sample['image_data'],
                text_prompt=sample['text_prompt'],
                model=model,
                generation_config=generation_config,
                processor=processor,
                eval_llm=eval_llm,
                eval_llm_tokenizer=eval_llm_tokenizer,
                question=sample['question'],
                answer=sample['answer'],
                n_iterations=args.max_num_iterations,
                clip_embedder=None,  # Already computed
                use_difficulty_adaptive=False,  # Already computed
                rollout_limit=args.rollout_limit,
            )
            
            # Compute reward (1 if solved, 0 otherwise)
            reward = 1.0 if solution is not None else 0.0
            
            # Update MAB
            difficulty_mab.update(selected_arm, reward)
            
            # Save successful solutions
            if solution is not None:
                result_data = sample['data'].copy()
                result_data['solution'] = ''.join(solution.solution_steps)
                result_data['iters'] = n_iter
                result_data['difficulty_score'] = float(sample['difficulty'])
                result_data['selected_arm'] = int(selected_arm)
                if sample['vision_emb'] is not None:
                    result_data['vision_embedding'] = sample['vision_emb'].cpu().numpy().tolist()
                final_response.append(result_data)
            
            # Print stats periodically
            if (iteration + 1) % 20 == 0:
                print(f"\n--- Iteration {iteration + 1} Stats ---")
                for stat in difficulty_mab.get_arm_stats():
                    print(f"  Arm {stat['arm']} {stat['difficulty_range']}: "
                          f"{stat['count']} samples, avg reward: {stat['avg_reward']:.3f}")
                torch.cuda.empty_cache()
                
        except Exception:
            difficulty_mab.update(selected_arm, 0.0)  # Count as failure
            continue
    
    # Final statistics
    print("\n=== Final Difficulty-based MAB Statistics ===")
    for stat in difficulty_mab.get_arm_stats():
        print(f"Arm {stat['arm']} {stat['difficulty_range']}: "
              f"{stat['count']} samples, avg reward: {stat['avg_reward']:.3f}")

    df = pd.DataFrame(final_response)
    df.to_parquet(args.output_file, index=False, engine='pyarrow')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--eval_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_pths", type=str, nargs='+', default="None")
    parser.add_argument("--output_file", type=str, default="answer.jsonl")
    #parser.add_argument("--max_num_iterations", type=int, default=50)
    parser.add_argument("--max_num_iterations", type=int, default=5)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    
    # Vision embeddings and difficulty estimation
    parser.add_argument("--use-vision-embeddings", action="store_true", default=True,
                        help="Extract vision embeddings from Qwen2.5-VL for multimodal representation")
    parser.add_argument("--use-difficulty-adaptive", action="store_true", default=True,
                        help="Use model confidence to estimate difficulty and adapt iterations")
    
    # Difficulty-based MAB arguments
    parser.add_argument("--difficulty-bins", type=int, default=3,
                        help="Number of difficulty bins (arms) for MAB: 3=[easy,medium,hard], 5=[very easy,...,very hard]")
    parser.add_argument("--exploration-c", type=float, default=2.0,
                        help="UCB exploration constant for difficulty-based MAB")
    parser.add_argument("--rollout-limit", type=int, default=5,
                        help="Maximum number of rollout steps per MAB iteration")
    
    args = parser.parse_args()

    main(args)

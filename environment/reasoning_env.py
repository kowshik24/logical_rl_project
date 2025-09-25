import gym
from gym import spaces
import numpy as np
import torch
from .logic_puzzles import PropositionalLogicGenerator

class LogicalReasoningEnv(gym.Env):
    """Properly implemented reasoning environment with constrained generation"""
    
    def __init__(self, model, tokenizer, max_length=128):
        super().__init__()
        
        self.generator = PropositionalLogicGenerator()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Action space: vocabulary size
        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        self.observation_space = spaces.Dict({
            'input_ids': spaces.Box(low=0, high=tokenizer.vocab_size, shape=(max_length,), dtype=np.int32),
            'attention_mask': spaces.Box(low=0, high=1, shape=(max_length,), dtype=np.int32)
        })
        
        # Get True/False token IDs
        self.true_token_id = tokenizer.encode("True")[0]
        self.false_token_id = tokenizer.encode("False")[0]
        
        self.reset_episode()
        
    def reset_episode(self):
        """Reset internal episode state"""
        self.current_puzzle = None
        self.expected_answer = None
        self.values = None
        self.expression = None
        self.generated_tokens = []
        self.step_count = 0
        self.max_steps = 2  # Only need 1-2 tokens for True/False
        self.episode_count = 0  # Track episodes for curriculum learning
        
    def reset(self):
        """Reset environment with new puzzle"""
        self.reset_episode()
        
        level = 1  # Start with simple puzzles
        self.current_puzzle, self.expected_answer, self.values, self.expression = \
            self.generator.generate_puzzle_by_level(level)
        
        # Create prompt that encourages True/False answers
        prompt = f"{self.current_puzzle} Answer:"
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )
        
        self.current_input_ids = inputs['input_ids'][0].numpy()
        self.current_attention_mask = inputs['attention_mask'][0].numpy()
        
        return {
            'input_ids': self.current_input_ids.copy(),
            'attention_mask': self.current_attention_mask.copy()
        }
        
    def step(self, action):
        """Execute one step - generate one token"""
        self.step_count += 1
        
        # Add the action (token) to our sequence
        self.generated_tokens.append(action)
        
        # Decode the generated tokens to get the text
        generated_text = self.tokenizer.decode(self.generated_tokens, skip_special_tokens=True)
        
        # Check if we have a complete answer
        done = False
        reward = 0.0
        
        # Check for True/False in generated tokens
        if action == self.true_token_id:
            done = True
            reward = self.generator.verify_answer(self.current_puzzle, "True", self.expected_answer)
        elif action == self.false_token_id:
            done = True
            reward = self.generator.verify_answer(self.current_puzzle, "False", self.expected_answer)
        elif self.step_count >= self.max_steps:
            done = True
            reward = -0.5  # Penalty for not giving True/False answer
        
        # Create new state (this is simplified - in practice you'd update the sequence)
        obs = {
            'input_ids': self.current_input_ids.copy(),
            'attention_mask': self.current_attention_mask.copy()
        }
        
        info = {
            'puzzle': self.current_puzzle,
            'generated_text': generated_text,
            'expected_answer': self.expected_answer,
            'generated_tokens': self.generated_tokens.copy()
        }
        
        return obs, reward, done, info
    

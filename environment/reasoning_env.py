import gym
from gym import spaces
import numpy as np
import torch
from .logic_puzzles import PropositionalLogicGenerator

class LogicalReasoningEnv(gym.Env):
    """Proper environment for logical reasoning with text generation"""
    
    def __init__(self, model, tokenizer, max_length=128):
        super().__init__()
        
        self.generator = PropositionalLogicGenerator()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Action space: vocabulary size for text generation
        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        
        # Observation space: encoded text with attention mask
        self.observation_space = spaces.Dict({
            'input_ids': spaces.Box(low=0, high=tokenizer.vocab_size, shape=(max_length,), dtype=np.int32),
            'attention_mask': spaces.Box(low=0, high=1, shape=(max_length,), dtype=np.int32)
        })
        
        self.current_puzzle = None
        self.expected_answer = None
        self.values = None  
        self.expression = None
        self.generated_text = ""
        self.current_step = 0
        self.max_steps = 20  # Maximum generation steps
        self.episode_count = 0  # Track episodes for curriculum learning
        
    def reset(self):
        """Reset the environment with a new puzzle"""
        self.current_step = 0
        self.generated_text = ""
        self.episode_count += 1
        
        # Start with simple puzzles always for now
        level = 1  
        self.current_puzzle, self.expected_answer, self.values, self.expression = \
            self.generator.generate_puzzle_by_level(level)
        
        # Create the prompt for the model
        prompt = self.current_puzzle + " Answer:"
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='np',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        self.current_input_ids = inputs['input_ids'][0]
        self.current_attention_mask = inputs['attention_mask'][0]
        
        return {
            'input_ids': self.current_input_ids.astype(np.int32),
            'attention_mask': self.current_attention_mask.astype(np.int32)
        }
        
    def step(self, action):
        """Execute one step - generate one token"""
        self.current_step += 1
        
        # Decode the action (token) and add to generated text
        try:
            new_token_text = self.tokenizer.decode([action], skip_special_tokens=False)
            self.generated_text += new_token_text
        except:
            # If decode fails, treat as invalid action
            reward = -0.1
            done = True
            info = {
                'puzzle': self.current_puzzle,
                'generated_text': self.generated_text,
                'expected_answer': self.expected_answer,
                'error': 'decode_failed'
            }
            return self._get_current_observation(), reward, done, info
        
        # Check if generation is complete (end token or max steps reached)
        done = False
        reward = 0.0
        
        # Check for completion conditions
        if (action == self.tokenizer.eos_token_id or 
            self.current_step >= self.max_steps or
            len(self.generated_text.strip()) > 10):  # Simple completion heuristic
            done = True
            # Evaluate the complete answer
            reward = self.generator.verify_answer(
                self.current_puzzle,
                self.generated_text.strip(),
                self.expected_answer
            )
        else:
            # Small positive reward for continuing generation appropriately
            reward = 0.01
        
        # Create new observation with the updated sequence
        obs = self._get_current_observation()
        
        info = {
            'puzzle': self.current_puzzle,
            'generated_text': self.generated_text,
            'expected_answer': self.expected_answer,
            'step': self.current_step
        }
        
        return obs, reward, done, info
    
    def _get_current_observation(self):
        """Get current observation with the generated text so far"""
        # Create the current text (original prompt + generated text)
        current_text = self.current_puzzle + " Answer:" + self.generated_text
        
        # Tokenize the current state
        inputs = self.tokenizer(
            current_text,
            return_tensors='np',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        return {
            'input_ids': inputs['input_ids'][0].astype(np.int32),
            'attention_mask': inputs['attention_mask'][0].astype(np.int32)
        }
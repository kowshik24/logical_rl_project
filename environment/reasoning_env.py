import gym
from gym import spaces
import numpy as np
from .logic_puzzles import PropositionalLogicGenerator

class LogicalReasoningEnv(gym.Env):
    """Custom environment for logical reasoning tasks"""
    
    def __init__(self, model, tokenizer, max_length=128):
        super().__init__()
        
        self.generator = PropositionalLogicGenerator()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Action space: vocabulary size for text generation
        self.action_space = spaces.Discrete(tokenizer.vocab_size)
        
        # Observation space: encoded text
        self.observation_space = spaces.Box(
            low=0, high=tokenizer.vocab_size, 
            shape=(max_length,), dtype=np.int32
        )
        
        self.current_puzzle = None
        self.current_step = 0
        self.max_steps = 10  # Maximum reasoning steps
        
    def reset(self):
        """Reset the environment with a new puzzle"""
        self.current_step = 0
        level = min(1 + (self.episode_count // 100), 5)  # Curriculum learning
        self.current_puzzle, self.expected_answer, _ = self.generator.generate_puzzle_by_level(level)
        
        # Encode the puzzle as input to the model
        encoded = self.tokenizer.encode(self.current_puzzle, return_tensors='np')
        return encoded[0]  # Return numpy array
        
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Generate answer using the model (simplified)
        # In reality, you'd use the model to generate text
        generated_answer = "True" if np.random.random() > 0.5 else "False"
        
        # Calculate reward
        reward = self.generator.verify_answer(self.current_puzzle, generated_answer)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or abs(reward) > 0.5
        
        # Return observation, reward, done, info
        # For simplicity, we return the same observation
        info = {"puzzle": self.current_puzzle, "answer": generated_answer}
        return self.reset() if done else self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation"""
        encoded = self.tokenizer.encode(self.current_puzzle, return_tensors='np')
        return encoded[0]
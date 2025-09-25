import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import numpy as np

class SimplePPOTrainer:
    """
    A simplified but functional PPO implementation for text generation
    """
    
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load a smaller model for faster experimentation
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)
        
        # PPO parameters
        self.epsilon = 0.1
        self.gamma = 0.99
        
    def get_action_and_value(self, observation):
        """Get action probabilities and value from current policy"""
        input_ids = torch.tensor(observation['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(observation['attention_mask']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Add numerical stability
            logits = torch.clamp(logits, min=-20, max=20)
            probs = torch.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample action
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[0, action] + 1e-8)
            
            # Simplified value estimate (could be improved with separate value head)
            value = torch.tensor(0.0).to(self.device)
        
        return action, log_prob.cpu().item(), value.cpu().item()
        
    def train_step(self, states, actions, rewards, log_probs_old):
        """Perform a single training step"""
        if len(states) == 0:
            return 0.0
            
        # Convert to tensors
        batch_input_ids = torch.tensor([s['input_ids'] for s in states]).to(self.device)
        batch_attention_mask = torch.tensor([s['attention_mask'] for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(self.device)
        
        # Get current policy's log probs
        outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits[:, -1, :]  # Last token logits
        
        # Add numerical stability
        logits = torch.clamp(logits, min=-20, max=20)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # Calculate ratio
        ratio = torch.exp(log_probs - log_probs_old)
        
        # Calculate advantages (simple version - could use GAE)
        if len(rewards) > 1:
            advantages = rewards - rewards.mean()
            advantages = advantages / (advantages.std() + 1e-8)
        else:
            advantages = rewards
        
        # PPO clipped loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (simplified - using rewards as targets)
        value_loss = torch.tensor(0.0).to(self.device)
        
        # Entropy bonus for exploration
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        entropy_bonus = -0.01 * entropy
        
        total_loss = policy_loss + value_loss + entropy_bonus
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
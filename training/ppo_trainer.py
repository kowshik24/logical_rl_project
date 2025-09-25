import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

class PPOTrainer:
    def __init__(self, model_name="microsoft/DialoGPT-small", learning_rate=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load a small pre-trained model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.epsilon = 0.2  # Clipping parameter
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
    def compute_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        """Compute advantages using GAE"""
        advantages = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            next_value = values[t]
            
        return torch.tensor(advantages)
    
    def ppo_update(self, states, actions, log_probs_old, rewards, values_old):
        """Perform PPO update step"""
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        log_probs_old = torch.stack(log_probs_old).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        values_old = torch.stack(values_old).to(self.device)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, values_old).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get new action probabilities and values
        outputs = self.model(input_ids=states, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]  # Last token logits
        new_log_probs = torch.log_softmax(logits, dim=-1)
        new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # PPO ratio and clipped objective
        ratio = torch.exp(new_log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values_old.squeeze(), rewards)
        
        # Entropy bonus
        entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
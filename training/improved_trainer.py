import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch.nn.functional as F

class ImprovedPPOTrainer:
    """
    Improved PPO trainer with balanced curriculum and better exploration
    """
    
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get True/False token IDs
        self.true_token_id = self.tokenizer.encode("True")[0]
        self.false_token_id = self.tokenizer.encode("False")[0]
        self.allowed_tokens = [self.true_token_id, self.false_token_id]
        
        print(f"True token ID: {self.true_token_id}, False token ID: {self.false_token_id}")
        
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)  # Lower learning rate
        
        # PPO parameters - more conservative
        self.epsilon = 0.15
        self.gamma = 0.99
        self.value_coef = 0.5
        self.entropy_coef = 0.05  # Higher entropy to encourage exploration
        
        # Value head
        self.value_head = nn.Linear(self.model.config.n_embd, 1).to(self.device)
        
        # Training statistics for balanced learning
        self.true_count = 0
        self.false_count = 0
        
    def get_action_and_value(self, state):
        """Get action and value with forced exploration"""
        input_ids = torch.tensor(state['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(state['attention_mask']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            
            # Constrain to True/False tokens only
            constrained_logits = torch.full_like(logits, -1e9)
            constrained_logits[:, self.allowed_tokens] = logits[:, self.allowed_tokens]
            
            # Add exploration bonus to less-used token
            if self.true_count > self.false_count + 10:
                # Encourage False if True is used too much
                constrained_logits[:, self.false_token_id] += 2.0
            elif self.false_count > self.true_count + 10:
                # Encourage True if False is used too much  
                constrained_logits[:, self.true_token_id] += 2.0
            
            # Temperature-based sampling with higher exploration
            probs = F.softmax(constrained_logits / 1.5, dim=-1)  # Higher temperature
            
            # Ensure minimum probability for exploration
            min_prob = 0.1
            probs[:, self.true_token_id] = torch.clamp(probs[:, self.true_token_id], min=min_prob)
            probs[:, self.false_token_id] = torch.clamp(probs[:, self.false_token_id], min=min_prob)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Track action usage
            if action.item() == self.true_token_id:
                self.true_count += 1
            else:
                self.false_count += 1
            
            # Get value estimate
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            value = self.value_head(hidden_state).squeeze()
            
        return action.item(), log_prob.item(), value.item()
    
    def train_step(self, states, actions, rewards, old_log_probs):
        """Training step with curriculum learning"""
        if len(states) == 0:
            return 0.0
        
        # Convert to tensors
        batch_input_ids = torch.stack([torch.tensor(s['input_ids']) for s in states]).to(self.device)
        batch_attention_mask = torch.stack([torch.tensor(s['attention_mask']) for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float).to(self.device)
        
        # Curriculum learning: Give higher weight to correct answers early on
        positive_rewards = rewards[rewards > 0]
        if len(positive_rewards) > 0:
            # Boost positive rewards to reinforce correct behavior
            rewards[rewards > 0] *= 2.0
        
        # Normalize rewards
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get current policy outputs
        outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]
        
        # Constrain logits
        constrained_logits = torch.full_like(logits, -1e9)
        constrained_logits[:, self.allowed_tokens] = logits[:, self.allowed_tokens]
        
        # Get current log probabilities
        log_probs = F.log_softmax(constrained_logits / 1.5, dim=-1)
        current_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get value estimates
        hidden_states = outputs.hidden_states[-1][:, -1, :]
        values = self.value_head(hidden_states).squeeze()
        
        # Ensure proper shapes
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        
        # Calculate advantages
        advantages = rewards - values.detach()
        
        # PPO loss with more conservative clipping
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, rewards)
        
        # Entropy bonus for exploration
        probs = F.softmax(constrained_logits / 1.5, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # Gradient clipping and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # More aggressive clipping
        self.optimizer.step()
        
        return total_loss.item()
    
    def get_stats(self):
        """Get training statistics"""
        total = self.true_count + self.false_count
        if total == 0:
            return "True: 0%, False: 0%"
        true_pct = self.true_count / total * 100
        false_pct = self.false_count / total * 100
        return f"True: {true_pct:.1f}%, False: {false_pct:.1f}%"
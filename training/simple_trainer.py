import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch.nn.functional as F

class SimplePPOTrainer:
    """
    A simplified but functional PPO implementation for text generation
    """
    
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get True/False token IDs for constrained generation
        self.true_token_id = self.tokenizer.encode("True")[0]
        self.false_token_id = self.tokenizer.encode("False")[0]
        self.allowed_tokens = [self.true_token_id, self.false_token_id, self.tokenizer.eos_token_id]
        
        print(f"True token ID: {self.true_token_id}, False token ID: {self.false_token_id}")
        
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        # PPO parameters
        self.epsilon = 0.2
        self.gamma = 0.99
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Simple value head (we'll use a linear layer on top of the model)
        self.value_head = nn.Linear(self.model.config.n_embd, 1).to(self.device)
        
    def get_action_and_value(self, state):
        """Get action and value from current policy"""
        # Convert state to tensor
        input_ids = torch.tensor(state['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(state['attention_mask']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Constrain to True/False tokens only
            constrained_logits = torch.full_like(logits, -1e9)
            constrained_logits[:, self.allowed_tokens] = logits[:, self.allowed_tokens]
            
            # Get probabilities
            probs = F.softmax(constrained_logits / 0.8, dim=-1)  # Temperature scaling
            
            # Sample action
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Get value estimate
            hidden_state = outputs.hidden_states[-1][:, -1, :]  # Last layer, last token
            value = self.value_head(hidden_state).squeeze()
            
        return action.item(), log_prob.item(), value.item()
        
    def train_step(self, states, actions, rewards, old_log_probs):
        """Perform a single training step"""
        if len(states) == 0:
            return 0.0
            
        # Convert to tensors more efficiently
        batch_input_ids = torch.stack([torch.tensor(s['input_ids']) for s in states]).to(self.device)
        batch_attention_mask = torch.stack([torch.tensor(s['attention_mask']) for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float).to(self.device)
        
        # Normalize rewards
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get current policy outputs
        outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]  # Last token logits
        
        # Constrain logits to allowed tokens
        constrained_logits = torch.full_like(logits, -1e9)
        constrained_logits[:, self.allowed_tokens] = logits[:, self.allowed_tokens]
        
        # Get current log probabilities
        log_probs = F.log_softmax(constrained_logits / 0.8, dim=-1)
        current_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get value estimates
        hidden_states = outputs.hidden_states[-1][:, -1, :]
        values = self.value_head(hidden_states).squeeze()
        
        # Calculate advantages
        advantages = rewards - values.detach()
        
        # PPO loss calculation
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, rewards)
        
        # Entropy bonus for exploration
        probs = F.softmax(constrained_logits / 0.8, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def generate_response(self, prompt, max_length=50):
        """Generate a response for debugging purposes"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
import os
import yaml
import torch
import numpy as np
from training.ppo_trainer import PPOTrainer
from environment.reasoning_env import LogicalReasoningEnv

def main():
    # Load configuration
    with open('training/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer and environment
    trainer = PPOTrainer(
        model_name=config['model_name'],
        learning_rate=float(config['learning_rate'])
    )
    
    env = LogicalReasoningEnv(trainer.model, trainer.tokenizer)
    
    # Training loop
    print("Starting RL training for logical reasoning...")
    
    for episode in range(config['total_episodes']):
        state = env.reset()
        episode_rewards = []
        states, actions, log_probs, rewards, values = [], [], [], [], []
        
        for step in range(config['max_steps_per_episode']):
            # Get action from policy
            with torch.no_grad():
                # Ensure state is properly converted to tensor
                if isinstance(state, np.ndarray):
                    state_tensor = torch.from_numpy(state).long().unsqueeze(0).to(trainer.device)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0).to(trainer.device)
                outputs = trainer.model(state_tensor)
                logits = outputs.logits[:, -1, :]
                
                # Add numerical stability - clip logits to prevent overflow
                logits = torch.clamp(logits, min=-20, max=20)
                
                # Apply temperature scaling for better exploration
                temperature = 1.0
                logits = logits / temperature
                
                probs = torch.softmax(logits, dim=-1)
                
                # Ensure probabilities are valid
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                action = torch.multinomial(probs, 1).item()
                log_prob = torch.log(probs[0, action] + 1e-8)  # Add small epsilon
                value = torch.tensor(0.5).to(trainer.device)  # Simplified value estimate
                
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            states.append(state_tensor.squeeze().cpu())
            actions.append(torch.tensor(action, dtype=torch.long))
            log_probs.append(log_prob.cpu())
            rewards.append(float(reward))
            values.append(value.cpu())
            
            state = next_state
            episode_rewards.append(reward)
            
            if done:
                break
        
        # Update policy using PPO
        if len(rewards) > 0:
            loss = trainer.ppo_update(states, actions, log_probs, rewards, values)
            
        if episode % 10 == 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.3f}, Loss: {loss:.3f}")

if __name__ == "__main__":
    main()
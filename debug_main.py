import torch
import numpy as np
from training.simple_trainer import SimplePPOTrainer
from environment.reasoning_env import LogicalReasoningEnv

def debug_training():
    """Debug version with extensive logging"""
    
    print("Initializing trainer and environment...")
    trainer = SimplePPOTrainer("gpt2")
    env = LogicalReasoningEnv(trainer.model, trainer.tokenizer)
    
    print("Starting training with debugging...")
    
    for episode in range(100):
        state = env.reset()
        episode_rewards = []
        states, actions, rewards, log_probs = [], [], [], []
        
        # Print first few puzzles for debugging
        if episode < 3:
            print(f"\nEpisode {episode}:")
            print(f"  Puzzle: {env.current_puzzle}")
            print(f"  Expected: {env.expected_answer}")
        
        step = 0
        total_reward = 0.0
        
        while step < env.max_steps:
            # Get action from current policy
            action, log_prob, value = trainer.get_action_and_value(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
            episode_rewards.append(reward)
            total_reward += reward
            step += 1
            
            if done:
                if episode < 3:
                    print(f"  Generated: '{info['generated_text']}'")
                    print(f"  Final reward: {reward}")
                break
        
        # Only update if we have meaningful rewards
        if len(rewards) > 0:
            loss = trainer.train_step(states, actions, rewards, log_probs)
        else:
            loss = 0.0
        
        # Extensive logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            positive_rewards = sum(1 for r in episode_rewards if r > 0)
            correct_answers = sum(1 for r in episode_rewards if r >= 1.0)
            
            print(f"Episode {episode:3d} | "
                  f"Total Reward: {total_reward:7.3f} | "
                  f"Avg Reward: {avg_reward:7.3f} | "
                  f"Positive: {positive_rewards}/{len(episode_rewards)} | "
                  f"Correct: {correct_answers} | "
                  f"Loss: {loss:7.3f}")
            
            # Print sample generation every 20 episodes
            if episode % 20 == 0 and episode > 0:
                print(f"  Last Generated: '{info.get('generated_text', '')}' -> Expected: '{info.get('expected_answer', '')}'")
                
        # Early success detection
        if episode > 10:
            recent_rewards = episode_rewards[-5:] if len(episode_rewards) >= 5 else episode_rewards
            if all(r >= 1.0 for r in recent_rewards) and len(recent_rewards) > 0:
                print(f"\nSUCCESS! Model getting consistent correct answers at episode {episode}")
                print(f"Last puzzle: {env.current_puzzle}")
                print(f"Generated: '{info['generated_text']}' -> Expected: '{info['expected_answer']}'")
                break

if __name__ == "__main__":
    debug_training()
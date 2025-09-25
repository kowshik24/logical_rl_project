import torch
import numpy as np
from training.simple_trainer import SimplePPOTrainer
from environment.reasoning_env import LogicalReasoningEnv

def debug_training():
    """Debug version with extensive logging"""
    
    print("Initializing trainer and environment...")
    trainer = SimplePPOTrainer("gpt2")
    env = LogicalReasoningEnv(trainer.model, trainer.tokenizer)
    
    print(f"True token: '{trainer.tokenizer.decode([trainer.true_token_id])}'")
    print(f"False token: '{trainer.tokenizer.decode([trainer.false_token_id])}'")
    print("Starting training with debugging...")
    
    successful_episodes = 0
    
    for episode in range(200):
        state = env.reset()
        episode_rewards = []
        states, actions, rewards, log_probs = [], [], [], []
        
        # Print first few puzzles for debugging
        if episode < 5 or episode % 50 == 0:
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
                if episode < 5 or episode % 50 == 0:
                    print(f"  Action token ID: {action}")
                    print(f"  Generated: '{info['generated_text']}'")
                    print(f"  Final reward: {reward}")
                    if reward > 0:
                        print(f"  ✓ CORRECT!")
                    else:
                        print(f"  ✗ WRONG")
                break
        
        # Track successful episodes
        if total_reward > 0:
            successful_episodes += 1
        
        # Only update if we have rewards
        if len(rewards) > 0:
            loss = trainer.train_step(states, actions, rewards, log_probs)
        else:
            loss = 0.0
        
        # Extensive logging
        if episode % 20 == 0:
            success_rate = successful_episodes / (episode + 1) * 100
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            positive_rewards = sum(1 for r in episode_rewards if r > 0)
            correct_answers = sum(1 for r in episode_rewards if r >= 1.0)
            
            print(f"Episode {episode:3d} | "
                  f"Success Rate: {success_rate:5.1f}% | "
                  f"Total Reward: {total_reward:7.3f} | "
                  f"Avg Reward: {avg_reward:7.3f} | "
                  f"Positive: {positive_rewards}/{len(episode_rewards)} | "
                  f"Loss: {loss:7.3f}")
                
        # Early success detection
        if episode > 50 and success_rate > 80:
            print(f"\nSUCCESS! Model achieving {success_rate:.1f}% success rate at episode {episode}")
            
            # Test the model on a few examples
            print("\nTesting model:")
            for test_episode in range(3):
                test_state = env.reset()
                action, _, _ = trainer.get_action_and_value(test_state)
                _, reward, _, info = env.step(action)
                print(f"  Test {test_episode+1}: {env.current_puzzle}")
                print(f"    Generated: '{info['generated_text']}' -> Expected: '{env.expected_answer}' -> {'✓' if reward > 0 else '✗'}")
            break

if __name__ == "__main__":
    debug_training()
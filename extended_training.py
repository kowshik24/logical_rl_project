import torch
import numpy as np
from training.simple_trainer import SimplePPOTrainer
from environment.reasoning_env import LogicalReasoningEnv

def extended_training():
    """Extended training with curriculum learning"""
    
    print("Starting extended training...")
    trainer = SimplePPOTrainer("gpt2")
    env = LogicalReasoningEnv(trainer.model, trainer.tokenizer)
    
    print(f"True token: '{trainer.tokenizer.decode([trainer.true_token_id])}'")
    print(f"False token: '{trainer.tokenizer.decode([trainer.false_token_id])}'")
    
    successful_episodes = 0
    best_success_rate = 0.0
    
    for episode in range(500):  # Extended training
        state = env.reset()
        episode_rewards = []
        states, actions, rewards, log_probs = [], [], [], []
        
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
                break
        
        # Track successful episodes
        if total_reward > 0:
            successful_episodes += 1
        
        # Only update if we have rewards
        if len(rewards) > 0:
            loss = trainer.train_step(states, actions, rewards, log_probs)
        else:
            loss = 0.0
        
        # Detailed logging every 25 episodes
        if episode % 25 == 0:
            success_rate = successful_episodes / (episode + 1) * 100
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                improvement = "📈"
            else:
                improvement = "  "
                
            print(f"Episode {episode:3d} | Success: {success_rate:5.1f}% | Best: {best_success_rate:5.1f}% | Loss: {loss:7.3f} {improvement}")
            
            # Show current example
            if episode > 0:
                print(f"    Last: {env.current_puzzle}")
                print(f"    Model: '{info.get('generated_text', '')}' -> Expected: '{env.expected_answer}' -> {'✓' if total_reward > 0 else '✗'}")
        
        # Success milestones
        if episode > 50:
            current_rate = successful_episodes / (episode + 1) * 100
            
            if current_rate >= 85 and episode > 100:
                print(f"\n🎉 EXCELLENT! Achieved {current_rate:.1f}% success rate!")
                break
            elif current_rate >= 75 and episode > 150:
                print(f"\n✨ VERY GOOD! Achieved {current_rate:.1f}% success rate!")
                break
    
    # Final comprehensive test
    print(f"\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    test_correct = 0
    test_total = 20
    
    print("Testing on fresh examples:")
    for i in range(test_total):
        test_state = env.reset()
        action, _, _ = trainer.get_action_and_value(test_state)
        _, reward, _, info = env.step(action)
        
        if reward > 0:
            test_correct += 1
        
        if i < 5:  # Show first 5 tests
            status = "✓" if reward > 0 else "✗"
            print(f"  {i+1}. {env.current_puzzle}")
            print(f"     Answer: {info['generated_text']} (Expected: {env.expected_answer}) {status}")
    
    final_accuracy = test_correct / test_total * 100
    print(f"\nFinal Test Accuracy: {final_accuracy:.1f}% ({test_correct}/{test_total})")
    
    if final_accuracy >= 80:
        print("🏆 Model has achieved strong logical reasoning performance!")
    elif final_accuracy >= 70:
        print("👍 Model shows good logical reasoning ability!")
    else:
        print("📚 Model needs more training to improve logical reasoning.")

if __name__ == "__main__":
    extended_training()
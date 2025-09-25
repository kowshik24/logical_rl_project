import torch
import numpy as np
from training.improved_trainer import ImprovedPPOTrainer
from environment.reasoning_env import LogicalReasoningEnv

def balanced_training():
    """Balanced training with curriculum learning to fix bias issues"""
    
    print("Starting balanced training to fix True bias...")
    trainer = ImprovedPPOTrainer("gpt2")
    env = LogicalReasoningEnv(trainer.model, trainer.tokenizer)
    
    print(f"True token: '{trainer.tokenizer.decode([trainer.true_token_id])}'")
    print(f"False token: '{trainer.tokenizer.decode([trainer.false_token_id])}'")
    print("="*60)
    
    # Create balanced training examples
    def create_balanced_examples():
        """Create examples with balanced True/False answers"""
        examples = []
        
        # True examples
        examples.extend([
            ("Given: A=True, B=True. What is A AND B? Answer True or False.", "True"),
            ("Given: A=True, B=True. What is A OR B? Answer True or False.", "True"), 
            ("Given: A=True, B=False. What is A OR B? Answer True or False.", "True"),
            ("Given: A=False, B=True. What is A OR B? Answer True or False.", "True"),
        ])
        
        # False examples  
        examples.extend([
            ("Given: A=False, B=False. What is A AND B? Answer True or False.", "False"),
            ("Given: A=False, B=False. What is A OR B? Answer True or False.", "False"),
            ("Given: A=True, B=False. What is A AND B? Answer True or False.", "False"),
            ("Given: A=False, B=True. What is A AND B? Answer True or False.", "False"),
        ])
        
        return examples
    
    balanced_examples = create_balanced_examples()
    successful_episodes = 0
    
    # Phase 1: Supervised learning on balanced examples
    print("Phase 1: Learning basic True/False patterns...")
    for epoch in range(50):
        epoch_correct = 0
        epoch_total = 0
        
        # Shuffle examples
        np.random.shuffle(balanced_examples)
        
        for puzzle, expected in balanced_examples:
            # Set up environment manually
            env.current_puzzle = puzzle
            env.expected_answer = expected
            
            # Reset and get initial state
            state = env.reset()
            env.current_puzzle = puzzle  # Override random puzzle
            env.expected_answer = expected
            
            # Get model prediction
            action, log_prob, value = trainer.get_action_and_value(state)
            _, reward, _, info = env.step(action)
            
            # Store for training
            states = [state]
            actions = [action]
            rewards = [reward]
            log_probs = [log_prob]
            
            # Train on this example
            loss = trainer.train_step(states, actions, rewards, log_probs)
            
            if reward > 0:
                epoch_correct += 1
            epoch_total += 1
        
        accuracy = epoch_correct / epoch_total * 100
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Accuracy {accuracy:5.1f}% | {trainer.get_stats()}")
            
        # Early stopping if achieving good accuracy
        if accuracy >= 90:
            print(f"✓ Phase 1 complete! Achieved {accuracy:.1f}% on balanced examples")
            break
    
    # Phase 2: Regular RL training with random puzzles
    print("\nPhase 2: Reinforcement learning on random puzzles...")
    
    for episode in range(200):
        state = env.reset()
        episode_rewards = []
        states, actions, rewards, log_probs = [], [], [], []
        
        step = 0
        total_reward = 0.0
        
        while step < env.max_steps:
            action, log_prob, value = trainer.get_action_and_value(state)
            next_state, reward, done, info = env.step(action)
            
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
        
        if total_reward > 0:
            successful_episodes += 1
        
        # Training
        if len(rewards) > 0:
            loss = trainer.train_step(states, actions, rewards, log_probs)
        else:
            loss = 0.0
        
        # Logging
        if episode % 25 == 0:
            success_rate = successful_episodes / (episode + 1) * 100
            
            print(f"Episode {episode:3d} | Success: {success_rate:5.1f}% | {trainer.get_stats()} | Loss: {loss:6.3f}")
            
            if episode > 0:
                status = "✓" if total_reward > 0 else "✗"
                print(f"    Last: {env.current_puzzle}")
                print(f"    Model: '{info.get('generated_text', '')}' -> Expected: '{env.expected_answer}' {status}")
        
        # Success criteria
        if episode > 50 and success_rate >= 80:
            print(f"\n🎉 SUCCESS! Achieved {success_rate:.1f}% success rate!")
            break
    
    # Final comprehensive test
    print(f"\n" + "="*60)
    print("FINAL COMPREHENSIVE TEST")
    print("="*60)
    
    # Test on systematic cases
    test_cases = [
        ("True AND True", "True"),
        ("True AND False", "False"),
        ("False AND True", "False"), 
        ("False AND False", "False"),
        ("True OR True", "True"),
        ("True OR False", "True"),
        ("False OR True", "True"),
        ("False OR False", "False")
    ]
    
    print("Systematic Logic Tests:")
    systematic_correct = 0
    
    for case, expected in test_cases:
        if "AND" in case:
            val1_str, val2_str = case.split(" AND ")
            val1 = val1_str == "True"
            val2 = val2_str == "True"
            puzzle = f"Given: A={val1}, B={val2}. What is A AND B? Answer True or False."
        else:
            val1_str, val2_str = case.split(" OR ")
            val1 = val1_str == "True" 
            val2 = val2_str == "True"
            puzzle = f"Given: A={val1}, B={val2}. What is A OR B? Answer True or False."
        
        env.current_puzzle = puzzle
        env.expected_answer = expected
        
        state = env.reset()
        state['input_ids'] = env.current_input_ids.copy()
        state['attention_mask'] = env.current_attention_mask.copy()
        
        action, _, _ = trainer.get_action_and_value(state)
        _, reward, _, info = env.step(action)
        
        status = "✓" if reward > 0 else "✗"
        if reward > 0:
            systematic_correct += 1
            
        print(f"  {case:15} -> Expected: {expected:5} | Model: {info['generated_text']:5} | {status}")
    
    systematic_accuracy = systematic_correct / len(test_cases) * 100
    
    # Random test
    print(f"\nRandom Puzzle Tests:")
    random_correct = 0
    for i in range(10):
        state = env.reset()
        action, _, _ = trainer.get_action_and_value(state)
        _, reward, _, info = env.step(action)
        
        if reward > 0:
            random_correct += 1
        
        if i < 3:  # Show first 3
            status = "✓" if reward > 0 else "✗"
            print(f"  {i+1}. Expected: {env.expected_answer:5} | Model: {info['generated_text']:5} | {status}")
    
    random_accuracy = random_correct / 10 * 100
    
    print("="*60)
    print(f"RESULTS:")
    print(f"Systematic Logic Accuracy: {systematic_accuracy:.1f}% ({systematic_correct}/{len(test_cases)})")
    print(f"Random Puzzle Accuracy: {random_accuracy:.1f}% ({random_correct}/10)")
    print(f"Token Usage: {trainer.get_stats()}")
    
    if systematic_accuracy >= 80 and random_accuracy >= 70:
        print("🏆 EXCELLENT! Model has learned logical reasoning!")
    elif systematic_accuracy >= 60 and random_accuracy >= 60:
        print("👍 GOOD! Model shows solid logical understanding!")
    else:
        print("📚 Model needs more balanced training.")

if __name__ == "__main__":
    balanced_training()
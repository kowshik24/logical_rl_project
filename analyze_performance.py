import torch
import numpy as np
from training.simple_trainer import SimplePPOTrainer
from environment.reasoning_env import LogicalReasoningEnv

def analyze_model_performance():
    """Analyze what types of problems the model gets right vs wrong"""
    
    print("Loading trained model...")
    trainer = SimplePPOTrainer("gpt2")
    env = LogicalReasoningEnv(trainer.model, trainer.tokenizer)
    
    # Test on different problem types
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
    
    print("\nTesting model on systematic cases:")
    print("=" * 50)
    
    correct_count = 0
    total_count = 0
    
    for case_type, expected in test_cases:
        # Create 5 test instances of each type
        for trial in range(5):
            # Manually create puzzle of this type
            if "AND" in case_type:
                val1 = "True" in case_type.split(" AND ")[0]
                val2 = "True" in case_type.split(" AND ")[1] 
                puzzle = f"Given: A={val1}, B={val2}. What is A AND B? Answer True or False."
            else:  # OR
                val1 = "True" in case_type.split(" OR ")[0]
                val2 = "True" in case_type.split(" OR ")[1]
                puzzle = f"Given: A={val1}, B={val2}. What is A OR B? Answer True or False."
            
            # Set up environment manually
            env.current_puzzle = puzzle
            env.expected_answer = expected
            
            # Get model prediction
            state = env.reset()
            state['input_ids'] = env.current_input_ids.copy()
            state['attention_mask'] = env.current_attention_mask.copy()
            
            action, _, _ = trainer.get_action_and_value(state)
            _, reward, _, info = env.step(action)
            
            model_answer = info['generated_text']
            is_correct = reward > 0
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            if trial == 0:  # Show first example of each type
                status = "✓" if is_correct else "✗"
                print(f"{case_type:20} -> Expected: {expected:5} | Model: {model_answer:5} | {status}")
    
    accuracy = correct_count / total_count * 100
    print("=" * 50)
    print(f"Overall Test Accuracy: {accuracy:.1f}% ({correct_count}/{total_count})")
    
    # Test random cases
    print(f"\nTesting on 20 random puzzles:")
    print("=" * 30)
    
    random_correct = 0
    for i in range(20):
        state = env.reset()
        action, _, _ = trainer.get_action_and_value(state)
        _, reward, _, info = env.step(action)
        
        if reward > 0:
            random_correct += 1
            
        if i < 5:  # Show first 5
            status = "✓" if reward > 0 else "✗"
            print(f"{i+1:2d}. {env.current_puzzle}")
            print(f"    Model: {info['generated_text']} | Expected: {env.expected_answer} | {status}")
            
    print(f"\nRandom Test Accuracy: {random_correct/20*100:.1f}% ({random_correct}/20)")

if __name__ == "__main__":
    analyze_model_performance()
import random
import re
from typing import Tuple, List, Dict

class PropositionalLogicGenerator:
    """Generates propositional logic puzzles with actual verification"""
    
    def __init__(self):
        self.variables = ['A', 'B', 'C', 'D', 'E', 'F']
        self.operators = ['AND', 'OR', 'IMPLIES']
        
    def generate_simple_puzzle(self) -> Tuple[str, str, Dict[str, bool], str]:
        """Generate a basic puzzle (level 1)"""
        var1, var2 = random.sample(self.variables[:3], 2)
        operator = random.choice(['AND', 'OR'])
        
        # Random truth values
        val1 = random.choice([True, False])
        val2 = random.choice([True, False])
        
        values = {var1: val1, var2: val2}
        expression = f"{var1} {operator} {var2}"
        puzzle = f"Given: {var1}={val1}, {var2}={val2}. What is {expression}? Answer True or False."
        
        # Calculate answer
        if operator == 'AND':
            answer = val1 and val2
        else:  # OR
            answer = val1 or val2
            
        return puzzle, str(answer), values, expression
    
    def generate_medium_puzzle(self) -> Tuple[str, str, Dict[str, bool], str]:
        """Generate medium difficulty puzzle with negation (level 2)"""
        var1, var2, var3 = random.sample(self.variables[:4], 3)
        val1, val2, val3 = [random.choice([True, False]) for _ in range(3)]
        
        values = {var1: val1, var2: val2, var3: val3}
        expression = f"({var1} AND NOT {var2}) OR {var3}"
        puzzle = f"Given: {var1}={val1}, {var2}={val2}, {var3}={val3}. What is {expression}? Answer True or False."
        
        answer = (val1 and not val2) or val3
        return puzzle, str(answer), values, expression
    
    def generate_puzzle_by_level(self, level: int) -> Tuple[str, str, Dict[str, bool], str]:
        """Generate puzzle based on curriculum level"""
        if level == 1:
            return self.generate_simple_puzzle()
        elif level == 2:
            return self.generate_medium_puzzle()
        else:
            # More complex puzzles for higher levels
            return self.generate_complex_puzzle(level)
    
    def verify_answer(self, puzzle_text: str, model_answer: str, expected_answer: str) -> float:
        """Verify if the model's answer is correct. Returns reward."""
        try:
            # Clean the model's answer
            model_answer_clean = model_answer.strip().upper()
            expected_clean = expected_answer.strip().upper()
            
            # Extract True/False from model's response
            if 'TRUE' in model_answer_clean:
                model_bool = True
            elif 'FALSE' in model_answer_clean:
                model_bool = False
            else:
                return -0.5  # Penalize invalid format but not as much as wrong answer
            
            expected_bool = (expected_clean == 'TRUE')
            
            if model_bool == expected_bool:
                return 1.0  # Correct answer
            else:
                return -1.0  # Wrong answer
                
        except Exception as e:
            return -0.5  # Error in parsing
    
    def generate_complex_puzzle(self, level: int) -> Tuple[str, str, Dict[str, bool], str]:
        """Generate complex puzzles for higher levels"""
        num_vars = min(3 + level - 2, 6)  # Increase variables with level
        vars_used = random.sample(self.variables, num_vars)
        values = {var: random.choice([True, False]) for var in vars_used}
        
        # Create a more complex logical expression
        var1, var2, var3 = vars_used[:3]
        val1, val2, val3 = values[var1], values[var2], values[var3]
        
        expression = f"({var1} IMPLIES {var2}) AND (NOT {var3} OR {var1})"
        puzzle = f"Given: {var1}={val1}, {var2}={val2}, {var3}={val3}. What is {expression}? Answer True or False."
        
        # Calculate answer: (A -> B) AND (~C OR A)
        # A -> B is equivalent to ~A OR B
        implies_result = (not val1) or val2
        or_result = (not val3) or val1
        answer = implies_result and or_result
        
        return puzzle, str(answer), values, expression
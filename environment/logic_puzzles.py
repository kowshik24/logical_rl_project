import random
import sympy
from sympy.logic.boolalg import to_cnf, And, Or, Not
from typing import Tuple, List, Dict

class PropositionalLogicGenerator:
    """Generates propositional logic puzzles with increasing difficulty"""
    
    def __init__(self):
        self.variables = ['A', 'B', 'C', 'D', 'E', 'F']
        self.operators = ['AND', 'OR', 'IMPLIES']
        
    def generate_simple_puzzle(self) -> Tuple[str, str, bool]:
        """Generate a basic puzzle (level 1)"""
        var1, var2 = random.sample(self.variables[:3], 2)
        operator = random.choice(['AND', 'OR'])
        
        # Random truth values
        val1 = random.choice([True, False])
        val2 = random.choice([True, False])
        
        puzzle = f"Given: {var1} = {val1}, {var2} = {val2}. What is {var1} {operator} {var2}?"
        
        # Calculate answer
        if operator == 'AND':
            answer = val1 and val2
        else:  # OR
            answer = val1 or val2
            
        return puzzle, str(answer).upper(), answer
    
    def generate_medium_puzzle(self) -> Tuple[str, str, bool]:
        """Generate medium difficulty puzzle with negation (level 2)"""
        var1, var2, var3 = random.sample(self.variables[:4], 3)
        val1, val2, val3 = [random.choice([True, False]) for _ in range(3)]
        
        puzzle = f"Given: {var1} = {val1}, {var2} = {val2}, {var3} = {val3}. What is ({var1} AND NOT {var2}) OR {var3}?"
        
        answer = (val1 and not val2) or val3
        return puzzle, str(answer).upper(), answer
    
    def generate_puzzle_by_level(self, level: int) -> Tuple[str, str, bool]:
        """Generate puzzle based on curriculum level"""
        if level == 1:
            return self.generate_simple_puzzle()
        elif level == 2:
            return self.generate_medium_puzzle()
        else:
            # More complex puzzles for higher levels
            return self.generate_complex_puzzle(level)
    
    def verify_answer(self, puzzle_text: str, model_answer: str) -> float:
        """Verify if the model's answer is correct. Returns reward."""
        # Extract ground truth from puzzle (in real implementation, you'd parse this properly)
        try:
            # This is a simplified verification - you'd want more robust parsing
            if "True" in model_answer.upper() or "TRUE" in model_answer:
                model_bool = True
            elif "False" in model_answer.upper() or "FALSE" in model_answer:
                model_bool = False
            else:
                return -1.0  # Invalid answer format
                
            # In a full implementation, you'd recalculate the expected answer
            # For now, we'll return a placeholder
            return 1.0 if random.random() > 0.5 else -1.0  # Placeholder
            
        except:
            return -1.0  # Error in parsing
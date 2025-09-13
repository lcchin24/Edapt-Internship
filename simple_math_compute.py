import dspy
import os
from dotenv import load_dotenv
import yaml
from dspy import PythonInterpreter

# Load environment variables
load_dotenv('key.env')

# Get API key from environment variable
api_key = os.getenv('api_key')

if not api_key:
    raise ValueError("API key not found in environment variables. Please check your key.env file.")

# Initialize the language model with the API key from environment
lm = dspy.LM('openai/gpt-4.1-nano', api_key=api_key, max_tokens=1000)
dspy.configure(lm=lm)

# Define the signature for math problem solving
class MathProblem(dspy.Signature):
    """Solve a given function based on the two given numbers and give the thought process"""
    
    number1 = dspy.InputField(desc="First number or context")
    number2 = dspy.InputField(desc="Second number or context") 
    function = dspy.InputField(desc="The mathematical function to perform")
    
    thought = dspy.OutputField(desc="The thought process of the function")
    answer = dspy.OutputField(desc="The answer to the function")

# Create the predictor with Python interpreter tool
react = dspy.ReAct(MathProblem, tools=[PythonInterpreter()])

# Load test data
with open('math_tests.yaml', 'r') as file:
    test_data = yaml.safe_load(file)

# Test the model
for i, example in enumerate(test_data):
    print(f"\nExample {i+1}:")
    print(f"Input: {example['input']}")
    print(f"Expected: {example['expected']}")
    
    # Get prediction
    result = react(
        number1=example["input"]["number1"],
        number2=example["input"]["number2"],
        function=example["input"]["function"]
    )
    
    print(f"Result: {result}")
    print(f"{'🟢 PASSED' if result.answer == example['expected'] else '🔴 FAILED'}")

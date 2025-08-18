import dspy
import os
from dotenv import load_dotenv
import yaml
from dspy import PythonInterpreter

load_dotenv('key.env')

api_key = os.getenv('api_key')

if not api_key:
    raise ValueError("API key not found in environment variables. Please check your key.env file.")

lm = dspy.LM('openai/gpt-4.1-nano', api_key=api_key, max_tokens=1000)
dspy.configure(lm=lm)


class IdentifyTargets(dspy.Signature):
    """Identify numerical values from noisy text"""
    input_text = dspy.InputField(desc="Raw block of text or YAML")
    targets = dspy.InputField(desc="Keywords to identify")
    
    thought = dspy.OutputField(desc="thought process of identification")
    numbers = dspy.OutputField(desc="Keyword value pairs as dictionary type")

class PerformFunction(dspy.Signature):
    """Perform mathematical function on identified numerical targets from text"""
    values = dspy.InputField(desc="Keyword value pairs as dictionary type")
    function = dspy.InputField(desc="The mathematical function to perform")

    thought = dspy.OutputField(desc="The thought process of the function")
    answer = dspy.OutputField(desc="The answer to the function. Format rules: 1) Return only the numeric value, 2) Include commas for thousands (e.g., 1,000,000), 3) Round to 2 decimal places for percentages")


react_extract = dspy.ProgramOfThought(IdentifyTargets)
react_compute = dspy.ReAct(PerformFunction, tools=[PythonInterpreter()])

with open('updated_math_tests.yaml', 'r') as file:
    test_data = yaml.safe_load(file)

for i, example in enumerate(test_data):
    print(f"\nExample {i+1}:")
    print(f"Input: {example['input']}")
    print(f"Expected: {example['input']['expected']}")
    
    
    values = react_extract(
        input_text=example["input"]["input_text"],
        targets=example["input"]["targets"],
        )

    result = react_compute(
        values=values.values,
        function=example["input"]["function"]
    )
    
    print(f"Result: {result}")
    print(f"{'🟢 PASSED' if result.answer == example['input']['expected'] else '🔴 FAILED'}")




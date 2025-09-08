import dspy
import os
from dotenv import load_dotenv
import yaml
from dspy import PythonInterpreter

load_dotenv('key.env')

api_key = os.getenv('api_key')

if not api_key:
    raise ValueError("API key not found in environment variables. Please check your key.env file.")

lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, max_tokens=5000)
dspy.configure(lm=lm)


class IdentifyTargets(dspy.Signature):
    """Identify numerical values from noisy text"""
    input_text = dspy.InputField(desc="Raw block of text or YAML")
    targets = dspy.InputField(desc="Values to identify")
    
    thought = dspy.OutputField(desc="thought process of identification")
    numbers = dspy.OutputField(desc="Extract ONLY the specific numerical values associated with each target keyword. For segmented data, look for patterns of numbers sharing the same label and extract the corresponding values. Return as a clean dictionary with target names as keys and their associated numbers as values.")

class PerformFunction(dspy.Signature):
    """Perform mathematical function on identified numerical targets from text"""
    values = dspy.InputField(desc="Keyword value pairs as dictionary type")
    function = dspy.InputField(desc="The mathematical function to perform, don't use external libraries like NumPy")

    thought = dspy.OutputField(desc="The thought process of the function")
    answer = dspy.OutputField(desc="The answer to the function, found without numpy. Format rules: 1) Return only the numeric value, 2) Normalize decimals to 2 decimal places, whole numbers should have no decimals, don't round averages to whole numbers")


react_extract = dspy.ProgramOfThought(IdentifyTargets)
react_compute = dspy.ReAct(PerformFunction, tools=[PythonInterpreter()])

with open('json_updated_math_tests.yaml', 'r') as file:
    test_data = yaml.safe_load(file)

for i, example in enumerate(test_data):
    print(f"\nExample {i+1}:")
    print(f"Input: {example['input']}")
    print(f"Expected: {example['input']['expected']}")
    
    
    values = react_extract(
        input_text=example["input"]["input_text"],
        targets=example["input"]["targets"],
        )

    print(f"Extracted values: {values.numbers}")
    print(f"Values type: {type(values.numbers)}")

    result = react_compute(
        values=values.numbers,
        function=example["input"]["function"]
    )
    
    print(f"Result: {result}")
    print(f"{'🟢 PASSED' if result.answer == example['input']['expected'] else '🔴 FAILED'}")




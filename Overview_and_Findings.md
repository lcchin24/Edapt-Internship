# DSPy Mathematical Reasoning System

A two-step mathematical reasoning pipeline built with DSPy that extracts numerical values and performs complex mathematical computations using large language models.

## System Overview

This system began with a singular signature approach using one `MathProblem` signature with a ReAct Agent to perform simple math computations as seen in `simple_math_compute.py`

To improve the systems capabilities, `math_extract_compute.py` now implements a **two-step approach** for mathematical reasoning, adding on to the previous one:

1. **Step 1 - Value Extraction**: `IdentifyTargets` extracts numerical values from noisy, unstructured text or structured data
2. **Step 2 - Mathematical Computation**: `PerformFunction` executes mathematical operations using PythonInterpreter for accuracy

## Core Components

### DSPy Signatures

**IdentifyTargets**
- Extracts specific numerical values associated with target keywords
- Handles segmented data and scattered values throughout text
- Returns clean dictionary format with target names as keys

**PerformFunction** 
- Performs mathematical operations on extracted values
- Uses PythonInterpreter for computational accuracy
- Enforces consistent output formatting (2 decimal places, no unnecessary rounding)

### Tools Used

The system uses DSPy's modular approach with:
- **ProgramOfThought** for value extraction
- **ReAct** for mathematical computation with tool integration
- **PythonInterpreter** for accurate numerical calculations

### Supported Models
- **GPT-4**: Baseline performance assessment
- **GPT-5**: Advanced capability evaluation with improved reasoning


## Prompt Engineering Insights

**Precise Role Definition**
- Clear, focused task descriptions eliminated ambiguity
- Specific signature definitions guided model behavior

**Detailed Output Specifications**
- Explicit formatting rules prevented inconsistent outputs
- Structured dictionary format enabled reliable data handling

### Examples and Outcomes

#### Target Identification Stage
```
"Extract ONLY the specific numerical values associated with each target keyword. 
For segmented data, look for patterns of numbers sharing the same label and 
extract the corresponding values. Return as a clean dictionary with target 
names as keys and their associated numbers as values."
```
**Outcome**: Led model to achieve greater accuracy in value extraction

#### Computation Stage
```
"Return only the numeric value. Normalize decimals to 2 decimal places, 
whole numbers should have no decimals, don't round averages to whole numbers."
```
**Outcome**: Ensured consistent output formatting across all test cases.

## Testing

### Test Categories
- **Basic Tests**: Simple arithmetic and statistical operations
- **Financial Tests**: Real-world scenarios with noise (phone numbers, irrelevant data)
- **Stress Tests**: Complex multi-step calculations with array processing and normalization

### Models Tested
- **GPT-4**: Baseline performance assessment (Both mini & nano)
- **GPT-5**: Advanced capability evaluation

### YAML Format Used
```yaml
- input:
    input_text: "Raw text containing numerical data"
    targets: ["target1", "target2", "target3"]
    function: "Mathematical operation description"
  expected: "Expected result"
```
### Test Examples

### 1. Financial Analysis with Noise
**Input**: Text with scattered irrelevant numbers
```
"Total funding across projects was finalized at 2,400,000 USD even though 
early drafts had typos like 23, 4567, and 89012 scattered around. 
Net revenue this time was reported at 600,000 after deducting all 
operating costs."
```

**Task**: Calculate percent increase in net revenue compared to last year's 550,000

**Result**: `(600,000 - 550,000) / 550,000 * 100 = 9.09%` (Both models succeeded)

### 2. Multiple Arrays of Numbers
**Input**: Nested dictionaries with arrays full of values
```
input_text: 
      {
        "departments": {
          "sales": [1000, 1100, 1200, 1150, 1300, 1250, 1400, 1350, 1450, 
          1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 
          2000],
          "engineering": [500, 520, 540, 560, 580, 600, 620, 640, 660, 
          680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880]
        },
        "inventory": {
            "widgets": [300, 320, 340, 360, 380, 400, 420, 440, 460, 480]
        }
      }
``` 

**Task**: Compute the total sum of all sales, engineering, and widgets values combined

**Result**: 48150 (Both Models Succeeded)

### 3. Scattered Target Extraction
**Input**: Quarterly data scattered throughout text
```
"Company X reported its 2025 quarterly financials as follows. Revenue streams 
across Q1 to Q4 (in millions) are: 125.5, 130.75, 128.3, and 140.9 respectively. 
Expenses are multi-faceted: Raw Materials (Q1 55.5, Q2 58.4, Q3 56.0, Q4 57.5), 
Labor costs (Q1 30.2, Q2 32.0, Q3 33.1, Q4 31.8)..."
```

**Task**: Calculate weighted average profit margin for qualifying quarters

**GPT-4 Result**: Failed to aggregate scattered quarterly data

**GPT-5 Result**: Successfully extracted and calculated weighted margin

### 4. Percentile Calculation
**Input**: Array of data
```
profits = [10, 20, 15, 25, 30, 40, 50, 35, 45, 55,65, 75, 85, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
```
**Task**: Calculate the sum of profits at or above the 80th percentile

**Results**: Returned incorrect answer due to attempting to use *numpy.percentile* which fails to load


## Findings


### GPT-4 Performance
**Strengths:**
- Perfect accuracy on extracting clear numerical targets in blocks of text
- Successful basic functions including conditional algebra and averaging
- Reliable performance on straightforward mathematical operations

**Limitations:**
- Failed to extract all parts of targets that were broken up and scattered throughout text
- Failed to correctly execute multi-step problems containing concepts like min-max normalization and weighted averaging
- Failed when trying to load external packages like NumPy as PythonInterpreter couldn't run them
- GPT-4o-nano failed further in multi-step calculations such as "'Compute average of data points, then subtract baseline. Round to 2 decimals.'"

### GPT-5 Performance
**Strengths:**
- Succeeded in all areas where GPT-4 performed well
- Correctly extracted and aggregated targets split up throughout text
- Correctly performed multi-step calculations that failed with GPT-4
- Better handling of complex mathematical concepts

**Limitations:**
- Still failed by trying to load external packages like NumPy as PythonInterpreter couldn't run them


### Common Failure Patterns (Both Models)

**Library Dependencies**
- PythonInterpreter couldn't run external packages like NumPy
- Impact: Statistical operations requiring numpy failed


## Conclusions

- **Model Evolution**: GPT-5 shows significant improvement over GPT-4 in complex reasoning
- **Prompt Engineering Success**: Detailed, structured prompts led to models achieving high accuracy on operations
- **Tool Integration Value**: PythonInterpreter eliminates computational errors but needs custom functions
- **Library Limitation**: External package dependencies remain a challenge for both models
- **Real-World Applicability**: Effective handling of noisy, unstructured financial data

### Future Research Directions
- Replace external library dependencies with custom functions 
- Explore pipelines outside of DSPy or PythonInterpreter that don't limit library access
- Test with functions in advanced mathematical domains like calculus, linear algebra, optimization



















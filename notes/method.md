# Self-Instruct Method Summary

## Overview
Self-instruct is a framework for generating instruction-following data using language models. The process creates a task pool of instructions and corresponding instances (input-output pairs) through iterative generation and filtering.

## Core Pipeline

### 1. Task Pool Initialization
- **Seed tasks**: 175 human-written instructions with 1 instance each
- **Format**: Each task contains an instruction and corresponding instance
- **Source**: [`seed_tasks.jsonl`](seed_tasks.jsonl) loaded via [`load_seed_tasks`](main.py:23-26)

### 2. Instruction Generation
**Process**: Sample from task pool → Generate new instructions → Filter → Add to pool

**Prompt Template** ([`new_instruction_prompt`](main.py:64-84)):
```
Come up with a series of tasks that are unique. There should be both classification and non-classification tasks. Follow after Task {N}. Follow the existing format with no changes or commentary.

Task 1: {existing_task}
Task 2: {existing_task}
...
Task 9: 
```

### 3. Task Classification
**Purpose**: Determine if each instruction is a classification task (finite output labels)

**Classification Prompt** ([`create_classification_prompt`](main.py:86-96)):
```
Can the following task be regarded as a classification task with finite output labels? Follow the provided format with no elaboration. Only return classification for Tasks that have a number.

Task: {instruction}
Is it classification? {Yes/No}
```

### 4. Instance Generation

#### Classification Tasks (Output-First)
**Approach**: Generate class labels first, then create matching inputs

**Prompt Template** ([`new_instruction_prompt`](main.py:81-82)):
```
Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate the correct class label.

Task: {instruction}
Class label: {label}
Input: {generated_input}
```

#### Non-Classification Tasks (Input-First)
**Approach**: Generate inputs first, then create corresponding outputs

**Prompt Template** ([`new_instruction_prompt`](main.py:78-80)):
```
Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

Task: {instruction}
Output: {generated_output}
```

## Implementation Details

### Key Components

#### Configuration Management
- **Location**: [`Config` class](main.py:13-21)
- **Parameters**:
  - `num_seed_tasks`: 8 (experimental setting)
  - `human_seed_ratio`: 0.8 (80% human-written seeds)
  - `max_iterations`: 1 (configurable)
  - `model_name`: "google/gemma-3n-e2b-it:free"

#### Sampling Strategy
- **Function**: [`sample_seeds_weighted`](main.py:110-131)
- **Bootstrap**: 100% human seeds for first iteration
- **Subsequent**: 80% human, 20% LLM-generated seeds

#### Pipeline Flow
1. **Initialization**: Load seed tasks from JSONL
2. **Sampling**: Select seed tasks based on ratio
3. **Generation**: Create new instructions via LLM
4. **Classification**: Determine task type
5. **Instance Generation**: Create input-output pairs
6. **Filtering**: Quality control (placeholder for future implementation)
7. **Integration**: Add new tasks to pool

### Figures and Tables

**Figure 1**: Pipeline Overview
```
Seed Tasks → Sampling → Instruction Generation → Classification → Instance Generation → Filtering → Task Pool
```

**Figure 2**: Classification Decision Tree
```
Task Instruction → Classification Check → [Yes: Output-First Generation] / [No: Input-First Generation]
```

**Table 1**: Task Type Distribution
| Iteration | Human Seeds | LLM Seeds | Classification | Non-Classification |
|-----------|-------------|-----------|----------------|-------------------|
| 0         | 8           | 0         | TBD            | TBD               |
| 1         | 6           | 2         | TBD            | TBD               |

## Key Design Decisions

1. **Modular Prompt Templates**: Centralized in [`LM` class](main.py:40-63) for consistency
2. **Configurable Ratios**: Dynamic sampling based on available LLM-generated tasks
3. **Bootstrap Mechanism**: Ensures quality initial generation with human seeds
4. **Separate Processing**: Different generation strategies for classification vs non-classification tasks

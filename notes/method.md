n-r Summary of methods in the paper "Self-instruct..."-tf

## Figure 2: High Level Overview
1. Create 175 seed tasks with 1 instruction and 1 instance per task -- this goes to a __task pool__
2. Sample from the __task pool__ and prompt an LLM to generate instructions for __new tasks__
3. Filer low quality generations from **step 2**
4. Add high quality instructions to the task pool

## What do task pool items look like?

The __task pool__ has *instructions* such as "Give me a quote from a famous person on this topic" and **instances** with "Honesty if the first chapter in the book of wisdom" -Thomas Jefferson

After these prompts have been fed into an LLM, it generates *new tasks*

Then, an LLM identifies if the task is **classification** or **not classification**

If the task is classification, the input prompt is output-first, meaning the model decides what the content is, then generates it.

If it is not classification, then the prompt is input-first. The model generates outputs for a provided context.

Finally, the **instruction** , *instance* , and __label__ are filtered and added back into the task pool.

### The prompt for generating new instructions:

"Come up with a series of tasks:
Task 1: {human made task}
Task 2: ...
...
Task 9: 
"

### Prompt for classifying whether a task instruction is a classification task or not:

"
Can the following task be regarded as a classification task with finite output labels?

Task: Given my personality and the job, tell me if I would be suitable. 
Is it classification? Yes
Task:

Given a set of numbers, find all possible subsets that sum to a given number. 
Is it classification? No
Task:
{instruction for the target task}
"

### Prompt for input-first aproach of instance generation
The model is prompted to generate the instance first, then generate the corresponding output:

"
Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.
Task: Output:
Which exercises are best for reducing belly fat at home?
Lying Leg Raises
Leg In And Out
Plank
Side Plank
Sit-ups
Task:
{Instruction for the target task}
"

### Prompt for output-first approach of instance generation
The model is prompted to generate the class label first, then give the input which leads to that label

"
Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate the
correct class label.
Task: Classify the sentiment of the sentence into positive, negative, or mixed. Class label: mixed
Sentence: I enjoy the flavor of the restaurant but their service is too slow. Class label: Positive
Sentence: I had a great day today. The weather was beautiful and I spent time with friends. Class label:
Negative
Sentence: I was really disappointed by the latest superhero movie. I would not recommend it.
Task:
{instruction for the target task}
"

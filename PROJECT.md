# Project: Replication of self-instruct paper

## Goal
Replicate the synthetic data generation in the paper.

## Current State
- Working data loading (human prompts)
- Working task generation
- Working task classification
- Instance generation works, but need to verify it outputs what I expect
- Filtering under development

### Limitations
- The model generally biases towards regression tasks
- The outputs of instance generation have not been thoroughly debugged
- There is no testing at all

## Nice to haves
1. Testing for expected model outputs
- Creation of pool, classification tasks have a known output. Ensure the model is consistently using few-shot learning
2. Metrics for pool filtering
- Currently, pool filtering is a bit of a black-box. It might be nice to run ablations on pool filtering to see how quality improves

## Major Questions
1. How do I determine if a model output is 'diverse' or ''good''. I suppose the human examples ensure this. Similarity will, of course, remove prompts that are stale. Over time this leads to diversity in outputs
2. What does the finished pipeline look like?
- Input desired number of pool tasks, get that many.
- Manual inspection can be used to fine-tune the filtering process until it scales, at which point ablations may be necessary

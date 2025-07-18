n-r Summary of methods in the paper "Self-instruct..."-tf

## Figure 2: High Level Overview
1. Create 175 seed tasks with 1 instruction and 1 instance per task -- this goes to a __task pool__
2. Sample from the __task pool__ and prompt an LLM to generate instructions for __new tasks__
3. Filer low quality generations from **step 2**
4. Add high quality instructions to the task pool

## What do task pool items look like?

The __task pool__ has *instructions* such as "Give me a quote from a famous person on this topic"

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
Task: Given my personality and the job, tell me if I would be suitable. Is it classification?
Task:
Yes
Give me an example of a time when you had to use your sense of humor. Is it classification? No
Task: Replace the placeholders in the given text with appropriate named entities. Is it classification? No
Task: Fact checking
-
tell me if the statement is true, false, or unknown, based on your knowledge and common sense.
Is it classification? Yes
Task: Return the SSN number for the person.
Is it classification?
Task:
No
Detect if the Reddit thread contains hate speech.
Is it classification? Yes
Task: Analyze the sentences below to identify biases.
Is it classification? No
Task: Select the longest sentence in terms of the number of words in the paragraph, output the sentence index.
Is it classification?
Yes
Task: Find out the toxic word or phrase in the sentence.
Is it classification? No
Task: Rank these countries by their population.
Is it classification? No
Task: You are provided with a news article, and you need to identify all the categories that this article belongs to. Possible categories include: Music, Sports, Politics, Tech, Finance, Basketball, Soccer, Tennis, Entertainment, Digital Game, World News. Output its categories one by one, seperated by comma.
Is it classification?
Task:
Yes
Given the name of an exercise, explain how to do it.
Is it classification?
No
Task: Select the oldest person from the list.
Is it classification? Yes
Task: Find the four smallest perfect numbers.
Is it classification? No
Task: Does the information in the document supports the claim? You can answer "Support" or "Unsupport".
Is it classification?
Yes
Task: Create a detailed budget for the given hypothetical trip.
Is it classification? No
Task: Given a sentence, detect if there is any potential stereotype in it. If so, you should explain the stereotype. Else, output no.
Is it classification? No
Task:
To make the pairs have the same analogy, write the fourth word.
Is it classification? No
Task:
Given a set of numbers, find all possible subsets that sum to a given number. Is it classification? No
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
Task: Extract all the country names in the paragraph, list them separated by commas. Example 1
Paragraph: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favourably in the United States. Output: English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.
Task:
Converting 85 F to Celsius.
Output: 85°F = 29.44°C
Task: Sort the given list ascendingly. Example 1
List: [10, 92, 2, 5, -4, 92, 5, 101]
Output: [-4, 2, 5, 5, 10, 92, 92, 101]
Example 2
Input 2 List: [9.99, 10, -5, -1000, 5e6, 999]
-
Output: [-1000, -5, 9.99, 10, 999, 5e6]
Task:
Suggest a better and more professional rephrasing of the following sentence. Example 1
Sentence: This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you to consider other candidates.
Output: This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it. I would suggest that you look at other properties. Example 2
Sentence: Just so you know, we did an experiment last week and found really surprising results language model can improve itself!
Output: Our experiments last week demonstrated surprising results, proving that the language model can improve itself.
Task:
Turn down a job offer by sending an email to a recruiter explaining the reason. Output: Hi [Recruiter],
Thank you so much for the generous offer to join your team. As we discussed, I've admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I've decided to accept an offer at another company.
I would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.
Thanks again,
[Your Name]
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
Task: Given a dialogue, classify whether the user is satisfied with the service. You should respond with "Satisfied" or "Unsatisfied".
Class label:
Dialogue:
Satisfied
Agent: Thank you for your feedback. We will work to improve our service in the future. Customer: I am happy with the service you provided. Thank you for your help. Class label: Unsatisfied
Dialogue:
Agent: Sorry that we will cancel your order. You will get a refund within 7 business days. Customer: oh that takes too long. I want you to take quicker action on this.
Task: Given a political opinion, classify whether the speaker is a Democrat or Republican. Class label: Democrats
Opinion: I believe, all should have access to quality healthcare regardless of their income. Class label: Republicans Opinion:
I believe that people should be able to keep more of their hard-earned money and should not be taxed at high rates.
Task: Tell me if the following email is a promotion email or not. Class label: Promotion
Email: Check out our amazing new sale! We've got discounts on all of your favorite products. Class label: Not Promotion
Email: We hope you are doing well. Let us know if you need any help.
Task: Detect if the Reddit thread contains hate speech.
Class label: Hate Speech
Thread: All people of color are stupid and should not be allowed to vote.
Class label: Not Hate Speech
Thread: The best way to cook a steak on the grill.
Task: Does the document supports the claim? Answer with "Support" or "Unsupport". Class label: Unsupport
Document: After a record-breaking run that saw mortgage rates plunge to all-time lows and home prices soar to new highs, the U.S. housing market finally is slowing. While demand and price gains are cooling, any correction is likely to be a modest one, housing economists and analysts say. No one expects price drops on the scale of the declines experienced during the
Great Recession.
Claim: The US housing market is going to crash soon.
Class label: Support
Document: The U.S. housing market is showing signs of strain, with home sales and prices slowing in many areas. Mortgage rates have risen sharply in recent months, and the number of homes for sale is increasing. This could be the beginning of a larger downturn, with some economists predicting a potential housing crash in the near future.
Claim: The US housing market is going to crash soon.
Task: Which of the following is not an input type? (a) number (b) date (c) phone number (d) email address (e) all of these are valid inputs.
Class label: (e)
Task:
{instruction for the target task}
"

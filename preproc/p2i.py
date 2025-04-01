from datasets import load_dataset

# "examples" is a DatasetDict (train/validation/test)
data_examples = load_dataset("xlangai/BRIGHT", "examples")
# "documents" is a single Dataset (no splits)
# data_documents = load_dataset("xlangai/BRIGHT", "documents")

# # Create one dictionary for *all* documents, keyed by "id".
# doc_lookup = {}
# for subset in data_documents.values():
#     for doc in subset:
#         doc_id = doc["id"]
#         doc_text = doc["content"]
#         doc_lookup[doc_id] = doc_text

# subset = 'theoremqa_theorems'
# subset = 'aops'
subset = 'theoremqa_questions'
query = data_examples[subset]['query']

requests=[]

for i, q in enumerate(query):
    p1 = r"""A photographer is taking a series of portraits of runners in a marathon. The photographer stands at a fixed distance from the track at a height of 2 meters. As a runner approaches at a speed of 6 meters per second, the angle of elevation \( \theta \) from the photographer’s position to the runner changes. If the runner is currently 20 meters away from the photographer along the track, how fast is the angle of elevation \( \theta \) changing when the runner is at that distance?"""
    s1 = r"""Let \( x \) be the horizontal distance from the photographer to the runner, which is given as \( x = 20 \) meters. The height \( h \) from which the photographer is taking the picture is 2 meters. We want to find the rate of change of the angle \( \theta \) with respect to time, \( \frac{d\theta}{dt} \).

From the relationship of the triangle formed by the photographer, the runner, and the ground, we know:

\[
\tan(\theta) = \frac{h}{x} = \frac{2}{x}
\]

To find the rate of change of \( \theta \) with respect to time, we will differentiate both sides implicitly with respect to time \( t \):

\[
\sec^2(\theta) \frac{d\theta}{dt} = -\frac{2}{x^2} \frac{dx}{dt}
\]

Given that the runner is moving towards the photographer, \( \frac{dx}{dt} = -6 \) meters per second.

Next, calculate \( \theta \) when \( x = 20 \):

\[
\tan(\theta) = \frac{2}{20} = \frac{1}{10}
\]

Now we find \( \theta \):

\[
\theta = \tan^{-1}\left(\frac{1}{10}\right)
\]

To compute \( \sec^2(\theta) \):

\[
\sec^2(\theta) = 1 + \tan^2(\theta) = 1 + \left(\frac{1}{10}\right)^2 = 1 + \frac{1}{100} = \frac{101}{100}
\]

Now substitute back into the differentiated equation:

\[
\frac{101}{100} \frac{d\theta}{dt} = -\frac{2}{20^2} (-6)
\]

Calculate \( \frac{2}{20^2} \):

\[
\frac{2}{400} = \frac{1}{200}
\]

Thus, we have:

\[
\frac{101}{100} \frac{d\theta}{dt} = \frac{6}{200} = \frac{3}{100}
\]

Now solve for \( \frac{d\theta}{dt} \):

\[
\frac{d\theta}{dt} = \frac{3}{100} \cdot \frac{100}{101} = \frac{3}{101}
\]

The rate of change of the angle of elevation \( \theta \) when the runner is 20 meters away is:

\[
\frac{d\theta}{dt} = \frac{3}{101} \text{ radians per second.}
\]
"""

    p2 = r"""A fruit shop has an assortment of 5 types of fruit: apples, bananas, cherries, dates, and elderberries. A customer wants to create a fruit basket that contains exactly 10 pieces of fruit. However, the customer can select any number of each type of fruit, including not selecting any of a certain type. How many different ways can the customer create this fruit basket? To solve this problem, we need to determine how many different non-negative integer combinations of the fruit types will add up to a total of 10 pieces."""
    s2 = r"""Let \( x_1 \), \( x_2 \), \( x_3 \), \( x_4 \), and \( x_5 \) represent the number of apples, bananas, cherries, dates, and elderberries in the basket, respectively. We need to find the number of solutions to the equation:

\[
x_1 + x_2 + x_3 + x_4 + x_5 = 10
\]

where \( x_i \geq 0 \) for \( i = 1, 2, 3, 4, 5 \).

This is a classic problem in combinatorics that can be solved using the stars and bars theorem. The theorem states that the number of ways to distribute \( n \) indistinguishable objects (in this case, fruits) into \( k \) distinguishable boxes (the different types of fruits) is given by the formula:

\[
\binom{n + k - 1}{k - 1}
\]

In our case, \( n = 10 \) (the number of pieces of fruit) and \( k = 5 \) (the different types of fruits). Applying the formula:

1. Calculate \( n + k - 1 = 10 + 5 - 1 = 14 \).
2. Calculate \( k - 1 = 5 - 1 = 4 \).

Now, we find the combination:

\[
\binom{14}{4} = \frac{14!}{4! \cdot (14 - 4)!} = \frac{14!}{4! \cdot 10!}
\]

Calculating \( \binom{14}{4} \):

\[
\binom{14}{4} = \frac{14 \times 13 \times 12 \times 11}{4 \times 3 \times 2 \times 1} = \frac{24024}{24} = 1001
\]

Thus, the total number of different ways the customer can create the fruit basket is 

\[
\boxed{1001}
\]
"""

    prompt=f"""Your task is to analyze a math problem and identify the key theorem for solving the math question.

Here are some examples:
**Problem**
{p1}
**Solution**
{s1}
**Theorem**
The Chain Rule in Calculus

**Problem**
{p1}
**Solution**
{s1}
**Theorem**
Formula for Combinations

Following the above format, write the solution after **Solution** and theorem name after **Theorem**. Your task is:
**Problem**
{q}
"""

    ls = {'aops':5000, 'theoremqa_theorems':5000,'theoremqa_questions':5000}

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    requests.append({
        "custom_id": f"request-{i+1}", 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": 
            {"model": "gpt-4o-mini", 
                "messages": messages,
            "max_tokens": ls[subset]},
    })

import json
file_path = f'{subset}_problems.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for request in requests:
        f.write(json.dumps(request) + '\n')

print(f"File saved to {file_path}")
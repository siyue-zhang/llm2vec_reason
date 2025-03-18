import torch
from llm2vec import LLM2Vec
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    # peft_model_name_or_path="/home/siyue/Projects/llm2vec_reason/output/mntp-supervised/Meta-Llama-3-8B-Instruct-mntp-simcse-e5-aug-scale+/E5Mix_train_m-Meta-Llama-3-8B-Instruct_p-mean_b-120_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    # peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

# Encoding queries using instructions
instruction = (
    # "Given a math equation, retrieve equivalent or similar equations:"
    # "Given a math formula name, retrieve relevant equations:"
    "Given a question, retrieve the right answer:"
)


queries = [
    [instruction, "2^2 + 4^2 = ?"],
    # [instruction, "$a^2 + b^2 = c^2$"],
    # [instruction, "Fundamental Theorem of Calculus"],
    # [instruction, "A ladder is leaning against a wall. The base of the ladder is 6 feet away from the wall, and the top of the ladder reaches a height of 8 feet from the ground. How long is the ladder?"],
]
q_reps = l2v.encode(queries)

# Encoding documents. Instruction are not required for documents
documents = [
#     'The length of the ladder is 100 feet.',
# """To solve for the length of the ladder, we use the Pythagorean Theorem: a^2 + b^2 = c^2.
# Let the distance from the wall (6 feet) be 'a', the height of the ladder (8 feet) be 'b', and the length of the ladder (which we need to find) be 'c'.

# a = 6
# b = 8

# We can plug these values into the Pythagorean Theorem formula:

# 6^2 + 8^2 = c^2
# 36 + 64 = c^2
# 100 = c^2

# Now, take the square root of both sides to find 'c':

# c = √100
# c = 10

# So, the length of the ladder is 10 feet.
# """,
    '6',
    '8',
    '10',
    '20',
    '22',
    '8^2',
    # "6^2+8^2=10^2",
    # "6^2+8^2=6.6^2",
    # "$a^2 + b^2 = c^2$"
    # "Fundamental Calculus",
    # "$a + b = c$",
    # "$a + b^2 = c$",
    # "\\frac{c^2}{a^2 + b^2} = 1",
    # "c = \sqrt{a^2 + b^2}",
    # "$c^2 - a^2 = b^2$",
    # "3^2 + 4^2 = 5^2",
    # "21^2 + 28^2 = 35^2",
    # "5^2-4^2=3^2",
    # "35^2-28^2=21^2",
    # "5*5 = \sqrt{3*3 + 4*4}",
    # "$a*a + b*b = c*c",
    # "$m^2 - n^2 - k^2 = 0$",
    # "$x^2 + y^2 = z^2$",
    # "$a^2 + b^2 = c^2$",
    # 'Pythagorean Theorem',
    # "Heron's Formula",
    # "Triangle Sum Theorem",
    # "Law of Sines"
    
#    "\int_{-3}^1 (7x^2 + x + 1)dx = F(1) - F(-3)",
#    "\int_\{2\}^\{5} (3x^3 - 4x + 2) \, dx = G(5) - G(2)",
#     "F'(x) = 3x^2 + 5x + 2 \\quad \\text{with\} \\quad F(x) = \\prod_{k=1}^{x} (3k^2 + 5k + 2)",
#     "\int_{-3}^1 (7x^2 + x +1)dx",
#     "What is \int_{-3}^1 (7x^2 + x +1)dx? To solve this integral, we first need to find the antiderivative of the given function, which is: F(x) = \int (7x^2 + x + 1)dx = (7/3)x^3 + (1/2)x^2 + x + C Now, we need to evaluate the definite integral from -3 to 1: \int_{-3}^1 (7x^2 + x + 1)dx = F(1) - F(-3) F(1) = (7/3)(1)^3 + (1/2)(1)^2 + (1) = 7/3 + 1/2 + 1 F(-3) = (7/3)(-3)^3 + (1/2)(-3)^2 + (-3) = -63 + 9/2 - 3 Now, subtract F(-3) from F(1): (7/3 + 1/2 + 1) - (-63 + 9/2 - 3) = (7/3 + 1/2 + 1) + (63 - 9/2 + 3) To add the fractions, we need a common denominator, which is 6: (14/6 + 3/6 + 6/6) + (378/6 - 27/6 + 18/6) = (23/6) + (369/6) = 392/6 Therefore, the answer is 392/6."
]
d_reps = l2v.encode(documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
for i in range(len(documents)):
    print(cos_sim[0][i], '  ', documents[i])
"""
tensor([[0.6470, 0.1619],
        [0.0786, 0.5844]])
"""
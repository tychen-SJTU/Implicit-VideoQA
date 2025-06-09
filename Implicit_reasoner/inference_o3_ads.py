from openai import OpenAI
import os
from PIL import Image
import base64
from io import BytesIO
import json
from tqdm import tqdm

api_key = ''
base_url = ''
client = OpenAI(api_key=api_key, base_url=base_url)


def request_gpt4v(prompt: str):
    response = client.chat.completions.create(
        model="o3",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "temperature": 0.1,
            }
        ],
    )
    return response.choices[0].message.content

# Load data
with open('./dataset/ads/converted_annotation.json', 'r') as f:
    data = json.load(f)

# Iterate over each item in data
answers = []
for item in tqdm(data):
    answer = {}
    video_id = str(item["video_id"])
    # question_id = str(item["question_id"])
    gt_answer = item['answer']
    clues = item["action_relation_intent"]
    # options = item['options']
    # print(options)
    question = ('Please choose one advertising strategy that align with the intent of the '
                'advertisement: (A) Social Identity, (B) Concreteness, (C) Anchoring and Comparison, '
                '(D) Overcoming Reactance, (E) Reciprocity, (F) Foot-in-the-Door, (G) Authority, (H) Social Impact,'
                ' (I) Anthropomorphism, (J) Scarcity, (K) Unclear.')


    clue_prompt_str = "\nContext clues are as following: \n"
    task_prompt2 = "\nBased on the clues, select the best option that accurately addresses the question.\n"
    question_prompt = 'Only give the best option.'
    # Generate action_relation_intent using GPT-4V
    prompt_case = question + clue_prompt_str + str(clues) + task_prompt2
    pred = request_gpt4v(prompt_case)

    answer["video_id"] = video_id
    answer["question"] = question
    answer["answer"] = gt_answer
    if pred is not None:
        if ')' in pred:
            pred = pred.split(")")[0]
        if '(' in pred:
            pred = pred.split("(")[-1]
        answer["pred"] = pred
        answers.append(answer)
        with open('/o3_ads.json', 'w') as f:
            json.dump(answers, f, indent=4)

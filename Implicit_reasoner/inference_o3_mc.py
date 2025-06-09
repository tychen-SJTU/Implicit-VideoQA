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

# base_folder = "/home/ICR/dataset/NExT-GQA/test_frames/nextgqa"

# Load data
with open('./dataset/ICR/test_mc.json', 'r') as f:
    data = json.load(f)

# Iterate over each item in data
answers = []
for item in tqdm(data):
    answer = {}
    video_id = str(item["video_id"])
    question_id = str(item["question_id"])
    gt_answer = item['ans']
    clues = item["action_relation_intent"]
    options = item['options']
    question = 'Please answer: ' + item['qa']['question'].capitalize() + '? '
    if len(options) == 5:
        question = ('Q: ' + question + '\n(A) ' + options[0] + '\n(B) ' + options[1] + '\n(C) ' + options[2]
                    + '\n(D) ' + options[3] + '\n(E) ' + options[4])
    else:
        question = ('Q: ' + question + '\n(A) ' + options[0] + '\n(B) ' + options[1] + '\n(C) ' + options[2]
                    + '\n(D) ' + options[3])

    clue_prompt_str = "\nContext clues are as following: \n"
    task_prompt2 = "\nBased on the clues, select the best option that accurately addresses the question.\n"
    question_prompt = 'Only give the best option.'

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
        with open('./XXX.json', 'w') as f:
            json.dump(answers, f, indent=4)

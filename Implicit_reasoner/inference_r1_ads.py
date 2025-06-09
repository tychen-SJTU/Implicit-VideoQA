from openai import OpenAI
import os
from PIL import Image
import base64
from io import BytesIO
import json
from tqdm import tqdm

api_key = ''
client = OpenAI(api_key=api_key, base_url="")
# prompt_path = 'prompt_action.txt'
#
# # Load prompt content
# with open(prompt_path, encoding="utf-8") as f:
#     task = f.readlines()
# task = ''.join(task)
# prompt = ("The question involves implicit visual information, with key visual evidence being invisible, "
#           "requiring the deduction of answer based on context. ")



def request_gpt4v(prompt: str):
    response = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return response.choices[0].message.content


# base_folder = "/mnt/sdb/dataset/persuasion/gpt_frames"
# Load data
with open('./dataset/ads/converted_annotation.json', 'r') as f:
    data = json.load(f)

# Iterate over each item in data
answers = []
for item in tqdm(data):
    answer = {}
    video_id = item['video_id']
    gt_answer = item['answer']
    clues = item["action_relation_intent"]
    question = ('Please choose one advertising strategy that align with the intent of the '
                'advertisement: (A) Social Identity, (B) Concreteness, (C) Anchoring and Comparison, '
                '(D) Overcoming Reactance, (E) Reciprocity, (F) Foot-in-the-Door, (G) Authority, (H) Social Impact,'
                ' (I) Anthropomorphism, (J) Scarcity, (K) Unclear.')

    clue_prompt_str = "Context clues are as following: \n"
    task_prompt2 = "\nBased on the clues, select the best option that accurately addresses the question.\n"
    # question_prompt = 'Only give the best option.'

    # Generate action_relation_intent using GPT-4V
    prompt_case = question + clue_prompt_str + str(clues) + task_prompt2
    pred = request_gpt4v(prompt_case)

    # answer["question_id"] = question_id
    answer["video_id"] = video_id
    answer["question"] = question
    answer["answer"] = gt_answer
    if pred is not None:
        answer["pred"] = pred.split("(")[-1].split(")")[0]
        answers.append(answer)
        with open('xxx.json', 'w') as f:
            json.dump(answers, f, indent=4)

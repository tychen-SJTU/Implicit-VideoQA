from openai import OpenAI
import os
from PIL import Image
import base64
from io import BytesIO
import json
from tqdm import tqdm

api_key = ''
client = OpenAI(api_key=api_key, base_url="")

prompt = ("The question involves implicit visual information, with key visual evidence being invisible, "
          "requiring the deduction of answer based on context. ")


def request_gpt4v(prompt: str):
    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        return response.choices[0].message
    except Exception as e:
        print(f"Error during GPT-4V request: {e}")
        return None


# Load data
with open('./dataset/ICR/test_open_ended.json', 'r') as f:
    data = json.load(f)

# Iterate over each item in data
answers = []
for item in tqdm(data):
    answer = {}
    video_id = str(item["video_id"])
    question_id = str(item["question_id"])

    question = 'Please answer: ' + item['qa']['question'].capitalize() + '? '
    gt_answer = item['ans']
    clues = item["action_relation_intent"]

    clue_prompt_str = "Context clues are as following: \n"

    prompt_case = prompt + clue_prompt_str + str(clues) + question
    pred = request_gpt4v(prompt_case)

    answer["question_id"] = question_id
    answer["video_id"] = video_id
    answer["question"] = question
    answer["answer"] = item['qa']["answer"]
    answer["pred"] = pred.content
    answers.append(answer)

    with open('./xxx.json', 'w') as f:
        json.dump(answers, f, indent=4)

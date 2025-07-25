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
prompt = ("The question involves implicit visual information, with key visual evidence being invisible, "
          "requiring the deduction of answer based on context. ")


def request_gpt4v(prompt: str, images: list, detail='auto'):
    try:
        response = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
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
with open('./dataset/ICR/test_mc.json', 'r', encoding='utf-8') as f:
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
    options = item['options']
    # print(options)
    if len(options) == 5:
        question = ('Q: ' + question + '\n(A) ' + options[0] + '\n(B) ' + options[1] + '\n(C) ' + options[2]
                    + '\n(D) ' + options[3] + '\n(E) ' + options[4])
    else:
        question = ('Q: ' + question + '\n(A) ' + options[0] + '\n(B) ' + options[1] + '\n(C) ' + options[2]
                    + '\n(D) ' + options[3])

    # Find corresponding video folder

    if video_id.startswith("v_"):
        video_name = f"{video_id}_anet_val{question_id}"
        video_folder = os.path.join("./test_frames/rextime", str(video_name))
    else:
        video_name = f"{video_id}_{question_id}"
        video_folder = os.path.join("./test_frames/nextgqa", str(video_name))

    if not os.path.exists(video_folder):
        continue

    # Collect all image paths in the folder
    image_paths = [os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.endswith(('.jpg', '.png'))]
    image_paths.sort()
    clue_prompt_str = "Context clues are as following: \n"
    task_prompt2 = "\nBased on the clues, select the best option that accurately addresses the question.\n"
    question_prompt = 'Only give the best option.'
    # Generate action_relation_intent using GPT-4V
    prompt_case = prompt + clue_prompt_str + str(clues) + task_prompt2 + question + question_prompt
    pred = request_gpt4v(prompt_case, image_paths).content

    answer["question_id"] = question_id
    answer["video_id"] = video_id
    answer["question"] = question
    answer["answer"] = gt_answer
    answer["pred"] = pred

    # if pred is not None:
    #     answer["pred"] = pred.split("(")[-1].split(")")[0]
    answers.append(answer)

    if pred is not None:
        if ')' in pred:
            pred = pred.split(")")[0]
        if '(' in pred:
            pred = pred.split("(")[-1]
        answer["pred"] = pred
        # answers.append(answer)
        with open('./xxx.json', 'w') as f:
            json.dump(answers, f, indent=4)

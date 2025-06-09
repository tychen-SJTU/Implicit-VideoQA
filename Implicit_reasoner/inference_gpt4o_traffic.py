from openai import OpenAI
import os
from PIL import Image
import base64
from io import BytesIO
import json
from tqdm import tqdm

api_key = ' '
base_url = ' '
client = OpenAI(api_key=api_key, base_url=base_url)
# prompt_path = 'prompt_action.txt'
#
# # Load prompt content
# with open(prompt_path, encoding="utf-8") as f:
#     task = f.readlines()
# task = ''.join(task)
prompt = ("The question involves implicit visual information, with key visual evidence being invisible, "
          "requiring the deduction of answer based on context. ")

def encode_image(image):
    if isinstance(image, str):
        image = Image.open(image)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes)
    img_str = img_base64.decode('utf-8')

    return img_str

def encode_image_gpt4v(image):
    return 'data:image/jpeg;base64,' + encode_image(image)

def request_gpt4v(prompt: str, images: list, detail='auto'):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] +
                           [{"type": "image_url", "image_url": {"url": encode_image_gpt4v(image), "detail": detail}}
                            for image in images],
                "temperature": 0.1,
            }
        ],
    )
    return response.choices[0].message.content

base_folder = "/home/ICR/dataset/NExT-GQA/test_frames/nextgqa"

# Load data
with open('./dataset/SUTD-traffic/filtered_data2.json', 'r') as f:
    data = json.load(f)

# Iterate over each item in data
answers = []
for item in tqdm(data):
    answer = {}
    video_id = str(item["video_id"])
    # question_id = str(item["question_id"])
    question = 'Please answer: ' + item['question'].capitalize() + '? '
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

    video_folder = os.path.join("/mnt/sdb/dataset/SUTD/gpt_frame", str(video_id[:-4]))

    if not os.path.exists(video_folder):
        continue

    # Collect all image paths in the folder
    image_paths = [os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.endswith(('.jpg', '.png'))]
    image_paths.sort()
    clue_prompt_str = "Context clues are as following: \n"
    task_prompt2 = "\nBased on the clues, select the best option that accurately addresses the question.\n"
    question_prompt = '\nOnly give the best option.'
    # Generate action_relation_intent using GPT-4V
    prompt_case = prompt + clue_prompt_str + str(clues) + task_prompt2 + question + question_prompt
    pred = request_gpt4v(prompt_case, image_paths)

    # answer["question_id"] = question_id
    answer["video_id"] = video_id
    answer["question"] = question
    answer["answer"] = gt_answer
    if pred is not None:
        answer["pred"] = pred.split("(")[-1].split(")")[0]
        answers.append(answer)
        with open('xxx.json', 'w') as f:
            json.dump(answers, f, indent=4)

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


with open('./dataset/ads/annotation.json', 'r') as f:
    data = json.load(f)

# Iterate over each item in data
answers = []
for item in tqdm(data.items()):
    answer = {}
    video_id = item[0]
    # question_id = str(item["question_id"])
    # question = 'Please answer: ' + item['qa']['question'].capitalize() + '? '
    gt_answer = item[1]
    clues = item["action_relation_intent"]
    # options = item['options']
    # print(options)
    question = ('Please choose one advertising strategy that align with the intent of the '
                'advertisement: (A) Social Identity, (B) Concreteness, (C) Anchoring and Comparison, '
                '(D) Overcoming Reactance, (E) Reciprocity, (F) Foot-in-the-Door, (G) Authority, (H) Social Impact,'
                ' (I) Anthropomorphism, (J) Scarcity, (K) Unclear.')

    # Find corresponding video
    video_folder = os.path.join("/mnt/sdb/dataset/persuasion/gpt_frames", str(video_id))

    # Collect all image paths in the folder
    image_paths = [os.path.join(video_folder, file) for file in os.listdir(video_folder) if
                   file.endswith(('.jpg', '.png'))]
    image_paths.sort()
    clue_prompt_str = "Context clues are as following: \n"
    task_prompt2 = "\nBased on the clues, select the best option that accurately addresses the question.\n"

    # Generate action_relation_intent using GPT-4V
    prompt_case = question + clue_prompt_str + str(clues) + task_prompt2
    pred = request_gpt4v(prompt_case, image_paths)

    # answer["question_id"] = question_id
    answer["video_id"] = video_id
    answer["question"] = question
    answer["answer"] = gt_answer
    answer["pred"] = pred
    answers.append(answer)
    with open('xxx.json', 'w') as f:
        json.dump(answers, f, indent=4)

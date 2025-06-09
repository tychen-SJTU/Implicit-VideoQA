import openai
from openai import OpenAI
import os
import argparse
import json
import ast
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default=r'', help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument("--api_base", default="", type=str, help="OpenAI API base.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")

    args = parser.parse_args(
        ['--pred_path', 'YYY.json',
         '--api_key', '',
         '--api_base', '',
         '--num_tasks', '4'
         ])

    return args


def main():
    args = parse_args()
    with open(args.pred_path, 'r') as f:
        new_pred_contents = json.load(f)
    api_key = ''
    base_url = ''
    client = OpenAI(api_key=api_key, base_url=base_url)

    response_list = []
    for sample in tqdm(new_pred_contents):
            question_id = sample['question_id']
            question = sample['question']
            answer = sample['answer']
            pred = sample['pred']
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ]
            )
            response_message = completion.choices[0].message.content.strip()
            response_message = ast.literal_eval(response_message)
            response_message['question'] = question
            response_message['YN'] = response_message['pred']
            response_message['pred'] = pred

            response_message['answer'] = answer
            response_message['question_id'] = question_id
            response_list.append(response_message)

            with open('/xxx.json', 'w') as f:
                json.dump(response_list, f, indent=4)

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for result in tqdm(response_list):
        try:
            # Computing score
            count += 1
            score_match = result['score']
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            pred = result['pred']
            if result["YN"] == "yes":
                yes_count += 1
            else:
                no_count += 1
        except:
            print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    main()


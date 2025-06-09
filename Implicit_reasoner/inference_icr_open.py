from utils.config import Config

config_file = "configs/config_mistral.json"
cfg = Config.from_file(config_file)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import io
from torch.utils.data import Dataset, DataLoader

from models.videochat_mistra.videochat2_it_mistral import VideoChat2_it_mistral
from utils.easydict import EasyDict
import torch
import json
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
import pandas
from PIL import Image
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop,
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from peft import get_peft_model, LoraConfig, TaskType
import copy

# load stage2 model
# cfg.model.vision_encoder.num_frames = 4
model = VideoChat2_it_mistral(config=cfg.model)

# add lora to run stage3 model
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=16, lora_alpha=32, lora_dropout=0.,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ]
)
model.mistral_model = get_peft_model(model.mistral_model, peft_config)

state_dict = torch.load("XXX.pth", "cpu")

if 'model' in state_dict.keys():
    msg = model.load_state_dict(state_dict['model'], strict=False)
else:
    msg = model.load_state_dict(state_dict, strict=False)
# print(msg)

model = model.eval()
model = model.cuda()
print('Load the VideoChat2 model')


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


def get_video_format(video_path):
    if os.path.exists(video_path + '.mp4'):
        video_path = video_path + '.mp4'
    elif os.path.exists(video_path + '.mkv'):
        video_path = video_path + '.mkv'
    elif os.path.exists(video_path + '.avi'):
        video_path = video_path + '.avi'
    elif os.path.exists(video_path + '.mov'):
        video_path = video_path + '.mov'
    elif os.path.exists(video_path + '.webm'):
        video_path = video_path + '.webm'
    else:
        video_path = None
    return video_path


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    # print(prompt)
    # print()
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
           repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([835]).to("cuda:0"),
        torch.tensor([2277, 29937]).to("cuda:0")]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            pad_token_id=model.mistral_tokenizer.eos_token_id,
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text[:-5]


def get_valid_frames(duration, vlen, num_frames, clips):
    frame_times = np.linspace(0, duration, vlen, endpoint=False)
    avoid_frames = set()
    for start, end in clips:
        start_idx = int(np.searchsorted(frame_times, start))
        end_idx = int(np.searchsorted(frame_times, end))
        avoid_frames.update(range(start_idx, end_idx))
    valid_frames = [i for i in range(vlen) if i not in avoid_frames]
    selected_frames = np.linspace(0, len(valid_frames) - 1, num_frames, dtype=int)
    selected_frame_indices = [valid_frames[i] for i in selected_frames]

    return selected_frame_indices


def load_video(video_path, duration, mask_duration, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    num_frames = len(vr)
    frame_indices = get_valid_frames(duration, num_frames, num_segments, mask_duration)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs


def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

    # print(f"n_position: {n_position}")
    # print(f"pre_n_position: {pre_n_position}")

    if n_position != pre_n_position:
        T = ckpt_num_frame  # checkpoint frame
        P = 14  # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5)  # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

    if cur_frame != ckpt_num_frame:
        # print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        # print(f'Interpolate the position embedding')
        T = ckpt_num_frame  # checkpoint frame
        new_T = cur_frame  # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5)  # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3)  # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

    return sinusoid_table


# Define Dataset class
class VideoQADataset(Dataset):
    def __init__(self, inference_file, video_clip_root, transform=None):
        with open(inference_file, 'r') as f:
            self.inference_list = json.load(f)
        self.video_clip_root = video_clip_root
        self.video_clip_root2 = "/mnt/sdb/dataset/Activitynet/v1-3/val"
        self.transform = transform

    def __len__(self):
        return len(self.inference_list)

    def __getitem__(self, idx):
        qasample = self.inference_list[idx]
        question = 'Please answer: ' + qasample['qa']['question'].capitalize() + '? '
        gt_answer = qasample['qa']['answer']
        video_id = qasample['video_id']
        mask_duration = qasample['duration']
        duration = qasample['all_duration']
        action_relation_intent = qasample["action_relation_intent"]
        video_id = str(video_id)
        if video_id.startswith("v_"):
            video_name = os.path.join(self.video_clip_root2, str(video_id))
        else:
            video_name = os.path.join(self.video_clip_root, str(video_id))

        video_path = get_video_format(video_name)
        vid, _ = load_video(video_path, duration, mask_duration, num_segments=8, return_msg=True,
                            resolution=224)
        #  open_ended
        # options = qasample['options']
        # print(options)
        # if len(options) == 5:
        #     question = ('Q: ' + question + '\n(A) ' + options[0] + '\n(B) ' + options[1] + '\n(C) ' + options[2]
        #                 + '\n(D) ' + options[3] + '\n(E) ' + options[4])
        # else:
        #     question = ('Q: ' + question + '\n(A) ' + options[0] + '\n(B) ' + options[1] + '\n(C) ' + options[2]
        #                 + '\n(D) ' + options[3])
        return {
            'question': question,
            'gt_answer': gt_answer,
            'vid': vid,
            'action_relation_intent': action_relation_intent,
            'question_id': int(qasample['question_id']),
            'video_id': (str(qasample['video_id'])),
            'origin_question': qasample['qa']['question']
        }


# Function to process batch
def process_batch(batch, model):
    results = []
    for sample in batch:
        pred_action_intent = sample["action_relation_intent"].split("\n")
        pred_action_intent = [item[3:] for item in pred_action_intent]
        action_list = [item.split(": ")[0] for item in pred_action_intent]
        intent_list = [item.split(": ")[1][:-1] for item in pred_action_intent]

        new_pos_emb = get_sinusoid_encoding_table(n_position=(224 // 16) ** 2 * 8, cur_frame=8, ckpt_num_frame=4)
        model.vision_encoder.encoder.pos_embed = new_pos_emb

        TC, H, W = sample["vid"].shape
        video = sample["vid"].reshape(1, TC // 3, 3, H, W).to("cuda:0")
        img_list = []
        cur_instruction = ("Extract parts of the contextual visual information that are beneficial for answering the "
                           "question: " + sample['origin_question'])
        with torch.no_grad():
            image_emb, _ = model.encode_img(video, cur_instruction)
        img_list.append(image_emb)

        action_emb_list, intent_emb_list = [], []
        for idx, action in enumerate(action_list):
            action_emb = model.mistral_tokenizer(action, return_tensors="pt", padding='max_length', truncation=True,
                                                 max_length=15).to(image_emb.device)
            inetnt_emb = model.mistral_tokenizer(intent_list[idx], return_tensors="pt", padding='max_length',
                                                 truncation=True,
                                                 max_length=15).to(image_emb.device)
            action_emb = model.mistral_model.base_model.model.model.embed_tokens(
                action_emb.input_ids)
            inetnt_emb = model.mistral_model.base_model.model.model.embed_tokens(
                inetnt_emb.input_ids)
            action_emb_list.append(action_emb.to(torch.float32))
            intent_emb_list.append(inetnt_emb.to(torch.float32))

        enhanced_intent = model.clue_enhance(action_emb_list, image_emb.squeeze(0).to(torch.float32))
        relation_pred = model.relation_head(intent_emb_list, enhanced_intent)
        relation_pred = [pred.squeeze(0)[0] for pred in relation_pred]
        result = [1 if t[1] + 0.1 > t[0] else 0 for t in relation_pred]

        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": " "
        })
        clue_prompt_str = ""
        if not (all((item == 0) for item in result) or (len(result) == 0)):
            clue_prompt_str = "Context clues are as following: "
            actions_for_refine_qformer = "Context key action clues: "
            last_index = len(result) - 1 - result[::-1].index(1)
            for label_idx, label in enumerate(result):
                if label == 1:
                    if label_idx != last_index:
                        actions_for_refine_qformer += action_list[label_idx] + ", "
                        clue_prompt_str += action_list[label_idx] + " to " + intent_list[
                            label_idx] + ", "
                    else:
                        actions_for_refine_qformer += action_list[label_idx] + ". "
                        clue_prompt_str += action_list[label_idx] + " to " + intent_list[
                            label_idx] + "."
        else:
            actions_for_refine_qformer = ""

        if actions_for_refine_qformer != "":
            img_embeds_refine, _ = model.encode_img(video, actions_for_refine_qformer)
            refined_output = model.vision_enhance(img_embeds_refine.to(torch.float32), img_list[0].to(torch.float32))
            img_list[0] = refined_output.to(torch.float16)

        task_prompt = ("The question involves some implicit visual information, with key visual evidence being "
                       "invisible, requiring the inference of the correct answer based on contextual "
                       "visual information and provided intention, action clues. ")
        # task_prompt2 = "Based on the clues, select the best option that accurately addresses the question.\n"
        # question_prompt = '\nOnly give the best option.'
        # answer_prompt = "Best option:("
        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
        chat.messages.append([chat.roles[0], task_prompt])
        if clue_prompt_str != '':
            chat.messages.append([chat.roles[0], clue_prompt_str])
        # chat.messages.append([chat.roles[0], task_prompt2])
        chat.messages.append([chat.roles[0], sample['question']])
        # chat.messages.append([chat.roles[0], question_prompt])

        output = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=100,
                        answer_prompt=None, print_res=False)
        output = output.split(")")[0]
        video_dict = {"video_id": sample["video_id"], "question_id": int(sample['question_id']),
                      "question": sample['question'], "answer": sample['gt_answer'], "pred": output}
        results.append(video_dict)

    return results


# Main script
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    video_clip_root = "/home/next/NExTVideo"
    inference_file = 'dataset/ICR/test_open_ended.json'
    output_file = 'XXX.json'

    dataset = VideoQADataset(inference_file, video_clip_root)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    all_results = []
    for batch in tqdm(dataloader):
        batch_list = [
            {key: value[idx] for key, value in batch.items()}
            for idx in range(len(batch[next(iter(batch.keys()))]))
        ]
        batch_results = process_batch(batch_list, model)
        all_results.append(batch_results)

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)

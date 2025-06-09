import logging
import os
import json
import random
import torch
import ast

import numpy as np

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.video_utils import VIDEO_READER_FUNCS

logger = logging.getLogger(__name__)


class ITImgTrainDataset_mistral(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(
            self, ann_file, transform,
            system="",
            start_token="<Image>", end_token="</Image>",
            random_shuffle=True,  # if True, shuffle the QA list
            return_question_instruction=False,  # if True, return instruction with instruciton
            dynamic_config=None,  # config for dynamic resolution finetuning
    ):
        super().__init__()

        # if len(ann_file) >= 3 and ann_file[2] == "video":
        #     self.media_type = "video"
        # else:
        #     self.media_type = "image"
        # self.label_file, self.data_root = ann_file[:2]
        self.media_type = "video"
        self.label_file = "dataset/ICR/training_5k.json"
        self.data_root = "/home/next/NExTVideo"
        self.data_root2 = "/mnt/sdb/dataset/Activitynet/v1-3/train"
        self.data_root3 = "/mnt/sdb/dataset/ETbench_icr/et"
        self.data_root4 = "/mnt/sdb/dataset/Activitynet/v1-3/val"

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform
        self.dynamic_config = dynamic_config
        if dynamic_config:
            logger.info(f"Finetuning with dynamic resolution: {dynamic_config}")

        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system."

        self.human_start = "[INST]"
        self.human_end = "[/INST]"
        self.assist_end = "</s>"
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.random_shuffle = random_shuffle
        # instruction location and number
        self.return_question_instruction = return_question_instruction
        logger.info(f"Random shuffle: {self.random_shuffle}")
        logger.info(f"Return question with instruction: {self.return_question_instruction}")

    def get_anno(self, index):
        file = self.anno[index]

        video_name = str(file["video_id"]) + ".mp4"
        video_name2 = str(file["video_id"]) + ".mkv"
        whole_duration = file["all_duration"]
        mask_duration = file["duration"]
        qa = file["qa"]
        context = file["context"]["context_question"]
        context = list(context.values())
        action_relation_intent = file["action_relation_intent"]
        action_relation_intent = action_relation_intent.split("\n")
        action_relation_intent = [relation[2:] for relation in action_relation_intent]
        action_label = file["action_mathing_label"]
        if "intent_mathing_label" in file:
            intent_label = file["intent_mathing_label"]
        else:
            intent_label = "[-1]"
        # Todo, 加载的时序问题
        if video_name.startswith('v_'):
            image_path = os.path.join(self.data_root2, video_name)
            if not os.path.exists(image_path):
                image_path = os.path.join(self.data_root2, video_name2)
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.data_root4, video_name)
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.data_root4, video_name2)
        elif "task" in file:
            image_path = os.path.join(self.data_root3, video_name)
        else:
            image_path = os.path.join(self.data_root, video_name)
        # if "start" in self.anno[index] and "end" in self.anno[index]:
        #     anno = {
        #         "image": os.path.join(self.data_root, filename), "qa": qa,
        #         "start": self.anno[index]["start"], "end": self.anno[index]["end"],
        #     }
        # else:
        anno = {"image": image_path, "qa": qa,
                "whole_duration": whole_duration, "mask_duration": mask_duration,
                "gt_action_relation_intent": context, "pred_action_relation_intent": action_relation_intent,
                "action_label": action_label, "intent_label": intent_label}
        return anno

    def __len__(self):
        return self.num_examples

    def process_qa(self, qa, msg=""):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        # if self.random_shuffle and len(qa) > 1:
        #     random.shuffle(qa)
        # if "i" in qa[0].keys() and qa[0]["i"] != "":
        #     cur_instruction = qa[0]["i"] + " "

        conversation = self.system
        # add instruction as system message
        if cur_instruction:
            conversation += cur_instruction

        # rstrip() for the extra " " in msg
        conversation += (
                self.human_start + " " + self.start_token + self.end_token + msg.rstrip() + " " + self.human_end
        )

        # for _, sentence in enumerate(qa):
        q = qa["question"]
        a = qa["answer"]
        if q != "":
            conversation += (" " + self.human_start + " " + q + " " + self.human_end)
        else:
            # no question, often in caption dataset
            pass
        conversation += (" " + a + " " + self.assist_end)

        # if self.return_question_instruction and cur_instruction:
        #     cur_instruction += qa[0]["q"]
        cur_instruction = ("Extract parts of the contextual visual information that are beneficial for answering the "
                           "question: ")
        if cur_instruction:
            cur_instruction += q
        return conversation.strip(), cur_instruction.strip()

    def __getitem__(self, index):
        try:
            # zou xia mian
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data_image(
                index, ann["image"],
                dynamic_config=self.dynamic_config
            )
            conversation, instruction = self.process_qa(ann["qa"])
            return image, conversation, instruction, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class ITVidTrainDataset_mistral(ITImgTrainDataset_mistral):
    media_type = "video"

    def __init__(
            self, ann_file, transform,
            num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
            system="", start_token="<Video>", end_token="</Video>",
            add_second_msg=False,
            random_shuffle=True,
            return_question_instruction=False,  # if True, return instruction with instruciton
            dynamic_config=None,  # config for dynamic resolution finetuning
    ):
        super().__init__(
            ann_file, transform,
            system=system,
            start_token=start_token, end_token=end_token,
            random_shuffle=random_shuffle,
            return_question_instruction=return_question_instruction,
            dynamic_config=dynamic_config,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg

        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            action_label = ann["action_label"]
            intent_label = ann["intent_label"]
            msg = ""
            clip = None
            # Todo 还得改
            # if "start" in ann and "end" in ann:
            clip = ann["mask_duration"]
            # MP4 -> video frames
            video, index, sec = self.load_and_transform_media_data_video(
                index, ann["image"], return_fps=True, clip=clip,
                dynamic_config=self.dynamic_config
            )
            ann["gt_action_relation_intent"] = [f"{' '.join(qa['question'].split(' ')[2:])}: {qa['answer']}"
                                                for qa in ann["gt_action_relation_intent"]]
            # gt_action_relation_intent = (ann["gt_action_relation_intent"][: 10] +
            #                                     [0] * (10 - len(ann["gt_action_relation_intent"])))
            # pred_action_relation_intent = (ann["pred_action_relation_intent"][: 10] +
            #                                       [0] * (10 - len(ann["pred_action_relation_intent"])))
            gt_action_relation_intent = '. '.join(ann["gt_action_relation_intent"])
            pred_action_relation_intent = ' '.join(ann["pred_action_relation_intent"])
            if self.add_second_msg and sec is not None:
                # " " should be added in the start and end
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
                # Todo process_QA
            conversation, instruction = self.process_qa(ann["qa"], msg)
            return (video, conversation, instruction,
                    gt_action_relation_intent, pred_action_relation_intent,
                    action_label, intent_label, index)
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class ITTextTrainDataset_mistral(ImageVideoBaseDataset):
    media_type = "text"

    def __init__(
            self, ann_file, transform,
            system="",
            start_token=None, end_token=None,
            random_shuffle=True,  # if True, shuffle the QA list
            return_question_instruction=False,  # if True, return instruction with instruciton
            dynamic_config=None,  # config for dynamic resolution finetuning
    ):
        super().__init__()
        self.media_type = "text"

        self.label_file, self.data_root = ann_file[:2]

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)

        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system."

        self.human_start = "[INST]"
        self.human_end = "[/INST]"
        self.assist_end = "</s>"
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.random_shuffle = random_shuffle
        # instruction location and number
        self.return_question_instruction = return_question_instruction
        logger.info(f"Random shuffle: {self.random_shuffle}")
        logger.info(f"Return question with instruction: {self.return_question_instruction}")

    def get_anno(self, index):
        qa = self.anno[index]["QA"]
        anno = {"qa": qa}
        return anno

    def __len__(self):
        return self.num_examples

    def process_qa(self, qa):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            cur_instruction = qa[0]["i"] + " "

        conversation = self.system
        # add instruction as system message
        if cur_instruction:
            conversation += cur_instruction

        for _, sentence in enumerate(qa):
            q = sentence["q"]
            a = sentence["a"]
            if q != "":
                conversation += (" " + self.human_start + " " + q + " " + self.human_end)
            else:
                # no question, often in caption dataset
                pass
            conversation += (" " + a + " " + self.assist_end)

        if self.return_question_instruction and cur_instruction:
            cur_instruction += qa[0]["q"]
        return conversation.strip(), cur_instruction.strip()

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            conversation, instruction = self.process_qa(ann["qa"])
            return torch.zeros(1), conversation, instruction, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

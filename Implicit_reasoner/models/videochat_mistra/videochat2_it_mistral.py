import random
import logging
import ast

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import CLIPTokenizer, CLIPTextModel
from ..blip2.blip2 import Blip2Base, disabled_train
from ..icr_modules import Causal_intent_RelationHead, Vision_clue_enhancement, Vision_action_enhancement
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class VideoChat2_it_mistral(Blip2Base):
    """
    VideoChat2 model.
    """

    def __init__(self, config):
        super().__init__()
        # pretrained_path
        vit_blip_model_path = config.get("vit_blip_model_path", None)
        mistral_model_path = config.get("mistral_model_path")
        videochat2_model_path = config.get("videochat2_model_path", "")
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        low_resource = config.get("low_resource", False)  # use 8 bit and put vit in cpu
        # qformer
        num_query_token = config.get("num_query_token")
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        self.qformer_text_input = config.get("qformer_text_input", False)
        # prompt
        max_txt_len = config.get("max_txt_len", 32)
        self.w_ce = config.get("w_ce", 3.0)
        self.human_start = "[INST]"
        self.human_end = "[/INST]"
        self.assist_end = "</s>"
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        logger.info(f"Add instruction in qformer: {self.qformer_text_input}")
        self.CE_loss = torch.nn.CrossEntropyLoss()
        # debug
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("/mnt/sdb/hg_model2/clip-vit-base-patch32")
        self.clip_model = CLIPTextModel.from_pretrained("/mnt/sdb/hg_model2/clip-vit-base-patch32")
        self.debug = config.get("debug", False)
        use_flash_attention = config.get("use_flash_attention", False)
        self.use_lora = config.get("use_lora", False)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)
        self.clip_matching = False
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.tokenizer.padding_side = "left"
        self.low_resource = low_resource
        self.vision_encoder, self.vision_layernorm = self.init_vision_encoder_umt(config)
        self.qformer, self.query_tokens = self.init_Qformer(
            num_query_token, config.vision_encoder.encoder_embed_dim,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )

        if not self.qformer_text_input:
            self.qformer.bert.embeddings.word_embeddings = None
            self.qformer.bert.embeddings.position_embeddings = None
            for layer in self.qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.qformer.resize_token_embeddings(len(self.tokenizer))
        self.qformer.cls = None

        if vit_blip_model_path:
            logger.info(f"Load ViT and QFormer from {vit_blip_model_path}")
            state_dict = torch.load(vit_blip_model_path, map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info('Loading ViT and Q-Former Done')

        self.extra_num_query_token = extra_num_query_token
        if extra_num_query_token > 0:
            logger.info(f"Add extra {extra_num_query_token} tokens in QFormer")
            self.extra_query_tokens = nn.Parameter(
                torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
            )

        if freeze_vit:
            logger.info("freeze vision encoder")
            for _, param in self.vision_encoder.named_parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train
            for _, param in self.vision_layernorm.named_parameters():
                param.requires_grad = False
            self.vision_layernorm = self.vision_layernorm.eval()
            self.vision_layernorm.train = disabled_train

        if freeze_qformer:
            logger.info("freeze Qformer")
            for _, param in self.qformer.named_parameters():
                param.requires_grad = False
            self.qformer = self.qformer.eval()
            self.qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        logger.info('Loading Mistral')
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)
        self.mistral_tokenizer.padding_side = "left"
        if not self.mistral_tokenizer.pad_token:
            logger.info("Set pad_token")
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

        if self.debug:
            logger.info("Debug mode, build small Mistral")
            mistral_config = AutoConfig.from_pretrained(mistral_model_path)
            mistral_config.hidden_size = 512
            mistral_config.intermediate_size = 2048
            mistral_config.num_attention_heads = 8
            mistral_config.num_hidden_layers = 12
            mistral_config.torch_dtype = torch.float16
            self.mistral_model = AutoModelForCausalLM.from_config(mistral_config)
        else:
            if use_flash_attention:
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_model_path,
                    torch_dtype=torch.float16,
                    # use_flash_attention_2=True,
                    attn_implementation="flash_attention_2",
                )
            else:
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_model_path,
                    torch_dtype=torch.float16,
                )

        logger.info("freeze Mistral")
        for _, param in self.mistral_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading Mistral Done')

        if self.use_lora:
            logger.info("Use lora to finetune mistral")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj", "lm_head"]
            )
            self.mistral_model = get_peft_model(self.mistral_model, peft_config)
            print("for mistral model:")
            self.mistral_model.print_trainable_parameters()

        self.relation_head = Causal_intent_RelationHead(config)
        self.clue_enhance = Vision_clue_enhancement(config)
        self.vision_enhance = Vision_action_enhancement(config)
        self.mistral_proj = nn.Linear(
            self.qformer.config.hidden_size, self.mistral_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len

        # load weights of VideoChat2
        if videochat2_model_path:
            logger.info(f"Load VideoChat2 from: {videochat2_model_path}")
            ckpt = torch.load(videochat2_model_path, map_location="cpu")
            if 'model' in ckpt.keys():
                msg = self.load_state_dict(ckpt['model'], strict=False)
            else:
                msg = self.load_state_dict(ckpt, strict=False)
            logger.info(msg)

        # causal_ckpt_path = '...'
        # causal_ckpt = torch.load(causal_ckpt_path, map_location="cpu")
        # relation_head_weights = {k: v for k, v in causal_ckpt["model"].items() if 'relation_head' in k}
        # relation_head_weights_new, clue_enhance_weights_new, vision_enhance_weights_new = {}, {}, {}
        # for key, value in relation_head_weights.items():
        #     new_key = key
        #     if 'relation_head.' in key:
        #         new_key = key.replace('relation_head.', '')  # 移除 relation_head 前缀
        #     relation_head_weights_new[new_key] = value
        # relation_head_msg = self.relation_head.load_state_dict(relation_head_weights_new, strict=False)
        #
        # clue_enhance_weights = {k: v for k, v in causal_ckpt["model"].items() if 'clue_enhance' in k}
        # for key, value in clue_enhance_weights.items():
        #     new_key = key
        #     if 'clue_enhance.' in key:
        #         new_key = key.replace('clue_enhance.', '')  # 移除 relation_head 前缀
        #     clue_enhance_weights_new[new_key] = value
        # clue_enhance_msg = self.clue_enhance.load_state_dict(clue_enhance_weights_new, strict=False)
        #
        # vision_enhance_weights = {k: v for k, v in causal_ckpt["model"].items() if 'vision_enhance' in k}
        # for key, value in vision_enhance_weights.items():
        #     new_key = key
        #     if 'vision_enhance.' in key:
        #         new_key = key.replace('vision_enhance.', '')  # 移除 relation_head 前缀
        #     vision_enhance_weights_new[new_key] = value
        # vision_enhance_msg = self.vision_enhance.load_state_dict(vision_enhance_weights_new, strict=False)
        # print("causal weight successfully loaded!!!")


    def vit_to_cpu(self):
        self.vision_layernorm.to("cpu")
        self.vision_layernorm.float()
        self.vision_encoder.to("cpu")
        self.vision_encoder.float()

    def encode_img(self, image, instruction):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            T = image.shape[1]
            use_image = True if T == 1 else False
            image = image.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]

            image_embeds = self.vision_encoder(image, use_image)
            B, T, L, C = image_embeds.shape
            image_embeds = image_embeds.reshape(B, -1, C)
            image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            if self.extra_num_query_token > 0:
                query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            else:
                query_tokens = self.query_tokens
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    instruction,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image_embeds.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                query_output = self.qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_mistral = self.mistral_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        return inputs_mistral, use_image

    def _get_text_len(self, text):
        return self.mistral_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def forward(self, image, text_input, instruction, gt_relation, pred_relation, action_label, intent_label, epoch):
        global action_list, intent_list, loss_ce
        img_embeds, use_image = self.encode_img(image, instruction)
        batch_size, img_len, _ = img_embeds.shape
        loss_ce = 0
        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []

        gt_pairs, pred_pairs = [], []
        for idx, gt_prompt in enumerate(gt_relation):
            gt_pairs.append(gt_prompt.split(". "))
            pred_pairs.append(pred_relation[idx].split(". "))

        # match the pairs while training
        # similarity matching which is not for backward propagation
        pairs_cross_examples_wo_backward = []
        for batch_id, gt_pair in enumerate(gt_pairs):
            pairs = []
            pred_pair = pred_pairs[batch_id]
            gt_actions = [pair.split(":")[0] for pair in gt_pair]
            pred_actions = [pair.split(":")[0] for pair in pred_pair]
            gt_actions = [s.lstrip() for s in gt_actions]
            pred_actions = [s.lstrip() for s in pred_actions]
            # similarity matching
            for pred_index, pred in enumerate(pred_actions):
                max_similarity = 0
                best_match_index = None
                self.clip_model = self.clip_model.to(img_embeds.device)

                pred_inputs = self.clip_tokenizer(pred, return_tensors="pt", padding=True, truncation=True).to(
                    img_embeds.device)
                with torch.no_grad():
                    pred_embedding = self.clip_model(**pred_inputs).last_hidden_state.mean(dim=1)

                for gt_index, gt in enumerate(gt_actions):
                    gt_inputs = self.clip_tokenizer(gt, return_tensors="pt", padding=True, truncation=True).to(
                        img_embeds.device)
                    with torch.no_grad():
                        gt_embedding = self.clip_model(**gt_inputs).last_hidden_state.mean(dim=1)
                    similarity = torch.nn.functional.cosine_similarity(pred_embedding, gt_embedding).item()
                    if similarity > 0.65 and similarity > max_similarity:
                        max_similarity = similarity
                        best_match_index = gt_index

                if best_match_index is not None:
                    pairs.append({
                        "pred_action": pred_pair[pred_index].split(": ")[0][1:].lstrip(),
                        "pred_intent": pred_pair[pred_index].split(": ")[1].lstrip() if pred_pair[pred_index].
                        split(": ")[1].lstrip().endswith(".") else pred_pair[pred_index].split(": ")[1].lstrip() + "."
                    })
            pairs_cross_examples_wo_backward.append(pairs)

        # gpt4_matching which is for backward propagation
        pairs_cross_examples = []
        for idx, gt_relation_sample in enumerate(gt_relation):
            # gt_relation_sample = gt_relation_sample.split(". ")
            pred_relation_sample = pred_relation[idx].split(". ")
            action_label_sample = ast.literal_eval(action_label[idx])
            # intent_label_sample = ast.literal_eval(intent_label[idx])
            # gt_actions = [relation.split(": ")[0] for relation in gt_relation_sample]
            # gt_intents = [relation.split(": ")[1] for relation in gt_relation_sample]
            pred_actions = [relation.split(": ")[0][1:] for relation in pred_relation_sample]
            # print(pred_relation_sample)
            # print(pred_relation_sample)
            # for sammple in pred_relation_sample:
            #     print(sammple.split(": ")[1])
            pred_intents = [relation.split(": ")[1] for relation in pred_relation_sample]
            # print(pred_intents)
            pred_actions = [s.lstrip() for s in pred_actions]
            pred_intents = [s.lstrip() for s in pred_intents]
            pairs = []
            for index, pred_action in enumerate(pred_actions):
                if action_label_sample[index] != -1:
                    pairs.append({
                        "pred_action": pred_action,
                        "pred_intent": pred_intents[index] if pred_intents[index].endswith(".")
                        else pred_intents[index] + "."
                    })
            pairs_cross_examples.append(pairs)

        # backward + non-backward -> complete pairs
        complete_pairs = [a + b for a, b in zip(pairs_cross_examples, pairs_cross_examples_wo_backward)]

        # calculate the text embeddings of all action-intent pairs
        all_action_list, all_intent_list = [], []
        for example_pairs in complete_pairs:
            action_list, intent_list = [], []
            for pair in example_pairs:
                pred_action = pair["pred_action"]
                pred_action_tokens = (
                    self.mistral_tokenizer(pred_action, return_tensors="pt", padding='max_length', truncation=True,
                                           max_length=15).
                    to(img_embeds.device))
                pred_intent = pair["pred_intent"]
                pred_intent_tokens = (
                    self.mistral_tokenizer(pred_intent, return_tensors="pt", padding='max_length', truncation=True,
                                           max_length=15).
                    to(img_embeds.device))

                if self.use_lora:
                    pred_action_embeds = self.mistral_model.base_model.model.model.embed_tokens(
                        pred_action_tokens.input_ids)
                    pred_intent_embeds = self.mistral_model.base_model.model.model.embed_tokens(
                        pred_intent_tokens.input_ids)
                else:
                    pred_action_embeds = self.mistral_model.model.embed_tokens(pred_action_tokens.input_ids)
                    pred_intent_embeds = self.mistral_model.model.embed_tokens(pred_intent_tokens.input_ids)

                action_list.append(pred_action_embeds)
                intent_list.append(pred_intent_embeds)

            all_action_list.append(action_list)
            all_intent_list.append(intent_list)

        # action_enhancement + relation deduction
        # calculate the pair relation
        # 0 for negative, 1 for positive
        self.clue_enhance = self.clue_enhance.to(img_embeds.device)
        self.relation_head = self.relation_head.to(img_embeds.device)

        results = []
        for index, actions in enumerate(all_action_list):
            if actions:
                intent_pred_query = all_intent_list[index]
                enhanced_intent = self.clue_enhance(actions, img_embeds[index, :, :])
                relation_pred = self.relation_head(intent_pred_query, enhanced_intent)
                relation_pred = [relation.squeeze(0)[0, :] for relation in relation_pred]
                relation_pred = [t.unsqueeze(0) for t in relation_pred]
                relation_pred = torch.cat(relation_pred, dim=0)
                result = [1 if t[1] > t[0] else 0 for t in relation_pred]
                results.append(result)
                # auxiliary loss calculation
                if intent_label[index] != '[-1]':
                    gt_list = ast.literal_eval(intent_label[index])
                    gt_length = len(gt_list)
                    gt_label = torch.tensor(gt_list).to(img_embeds.device)
                    loss_ce += self.CE_loss(relation_pred[:gt_length], gt_label)
            else:
                result = []
                results.append(result)
        text_input = list(text_input)
        actions_for_refine_qformers = []
        for index, result in enumerate(results):
            if not (all((item == 0) for item in result) or (len(result) == 0)):
                # ensuring there is at least one positive pair
                clue_prompt_str = "Context clues are as following: "
                actions_for_refine_qformer = "Context key action clues: "
                last_index = len(result) - 1 - result[::-1].index(1)  # find the last positive position index
                for label_idx, label in enumerate(result):
                    if label == 1:
                        clue_prompt_dict = complete_pairs[index][label_idx]
                        if label_idx != last_index:
                            # not the final pair
                            actions_for_refine_qformer += clue_prompt_dict["pred_action"] + ", "
                            clue_prompt_str += clue_prompt_dict["pred_action"] + " to " + clue_prompt_dict["pred_intent"][:-1] + ", "
                        else:
                            # the final pair
                            actions_for_refine_qformer += clue_prompt_dict["pred_action"] + "."
                            clue_prompt_str += clue_prompt_dict["pred_action"] + " to " + clue_prompt_dict["pred_intent"] + " "
                output_str = text_input[index].replace("[INST]", clue_prompt_str + "[INST]", 1)
                text_input[index] = output_str
            else:
                actions_for_refine_qformer = ""
            actions_for_refine_qformers.append(actions_for_refine_qformer)
        text_input = tuple(text_input)

        origin_img_embeds_list = list(torch.split(img_embeds, 1, dim=0))

        # enhance the vision input
        for idx, action in enumerate(actions_for_refine_qformers):
            if action != "":
                img_embeds_refine, _ = self.encode_img(image[idx].unsqueeze(0), action)
                refined_output = self.vision_enhance(img_embeds_refine, origin_img_embeds_list[idx])
                origin_img_embeds_list[idx] = refined_output
        img_embeds = torch.cat(origin_img_embeds_list, dim=0)

        # handle each prompt individually (written by videochat2.)
        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.mistral_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                tmp_img_embeds.device)
            p_after_tokens = self.mistral_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(
                tmp_img_embeds.device)
            if self.use_lora:
                p_before_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
            else:
                p_before_embeds = self.mistral_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.mistral_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.human_start + " "
            sep2 = " " + self.human_end + " "
            raw_text = p_after.split(sep2)
            for idx in range(0, len(raw_text) - 1):
                raw_text[idx] = raw_text[idx] + sep2
            # the first raw_text contains system and question
            # the last raw_text only contains answer
            # rstrip() for the extra " "
            answer_targets = p_after_tokens.input_ids.clone()
            # [target] "xxxxx. </s>"
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :cur_len] = -100
            for text in raw_text[1:-1]:
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]).rstrip())
                answer_targets[:, (cur_len + ans_len):(cur_len + total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())

            if self.debug:  # Inspect and check the correctness of masking
                z = answer_targets[0].clone()
                z = torch.where(z == -100, self.mistral_tokenizer.unk_token_id, z)
                logger.info(self.mistral_tokenizer.decode(z))

            assert cur_len == answer_targets.shape[
                1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {prompt}"

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)

        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(
            img_embeds.device) * self.mistral_tokenizer.pad_token_id
        if self.use_lora:
            inputs_embeds = self.mistral_model.base_model.model.model.embed_tokens(inputs_embeds)
        else:
            inputs_embeds = self.mistral_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        inputs_embeds[:, :1] = self.mistral_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len + 1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len + 1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len + img_len + 1):(input_len + 1)] = target_list[idx][0,
                                                                         :(input_len - p_before_len - img_len)]

        with self.maybe_autocast():
            outputs = self.mistral_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                use_cache=False,  # current flash_attn2 dows not support padding=right for mistral
            )

        return dict(
            loss=outputs.loss + (self.w_ce - 0.09 * epoch) * loss_ce,
        )

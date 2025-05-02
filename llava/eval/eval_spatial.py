import argparse
import copy
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()

    # Model
    model_path = os.path.expanduser(args.model_path)
    print(model_path)
    model_name = get_model_name_from_path(model_path)
    print("model name: ", model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(args.annotation_file) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions, total=len(questions)):
        question_id = line["id"]
        image_file = line["image_info"]["file_path"]
        text_question = line["text_q"]
        qa_info = line["qa_info"]
        print(os.path.join(args.image_folder, image_file))
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")

        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
        image_sizes = [image.size]

        conv = conv_templates[args.conv_mode].copy()
        conversations = line["conversations"]

        num_question = len(conversations) // 2
        for i in range(num_question):
            question = conversations[i * 2]["value"]

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            input_ids = input_ids.unsqueeze(0)

            stop_str = (
                conv_templates[args.conv_mode].sep
                if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
                else conv_templates[args.conv_mode].sep2
            )

            model.to(dtype=torch.bfloat16)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=128,
                    use_cache=True,
                )

            outputs = outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            ans_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "image": image_file,
                        "question": text_question,
                        "pred": outputs,
                        "gt": conversations[i * 2 + 1]["value"],
                        "model_id": model_name,
                        "qa_info": qa_info,
                    }
                )
                + "\n"
            )

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)

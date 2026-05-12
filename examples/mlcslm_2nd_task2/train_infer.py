import argparse
import json
import os
import re
import multiprocessing as mp
import torch
import warnings
import numpy as np

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def run_model(model, processor, messages, return_audio=False, use_audio_in_video=True,
              temperature=1e-2, top_p=0.1, top_k=1):
    from qwen_omni_utils import process_mm_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video
    )
    inputs = inputs.to(model.device).to(model.dtype)

    do_sample = temperature >= 0.01

    generate_kwargs = {
        "use_audio_in_video": use_audio_in_video,
        "return_audio": return_audio,
        "thinker_max_new_tokens": 8192,
        "max_new_tokens": 8192,
    }

    if do_sample:
        generate_kwargs.update({
            "do_sample": True,
            "thinker_do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
        if top_k > 0:
            generate_kwargs["top_k"] = top_k
    else:
        generate_kwargs.update({
            "do_sample": False,
            "thinker_do_sample": False,
        })

    output = model.generate(**inputs, **generate_kwargs)

    text_output = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = text_output[0] if text_output else ""

    return response, None


def extract_choice(response):
    if not response:
        return None
    patterns = [
        r'(?:answer|choice|option|答案|选项)[^A-Za-z]*([A-D])\b',
        r'\b([A-D])[\.\)、。:：]',
        r'(?:^|[\s\n])([A-D])(?=[\s\n\.\)、。:：]|$)',
        r'\b([A-D])\b',
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def worker(gpu_id, task_queue, result_queue, model_path, use_audio_in_video, return_audio):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    print(f"[GPU {gpu_id}] Model loaded successfully")

    while True:
        try:
            task = task_queue.get(timeout=2)
            if task is None:   # poison pill
                break
            idx, item = task
        except Exception:
            continue

        audio_path = item["audios"][0]

        user_msg = next(m for m in item["messages"] if m["role"] == "user")
        raw_content = user_msg["content"]
        if isinstance(raw_content, list):
            question_text = "".join(
                c.get("text", "") for c in raw_content if c.get("type") == "text"
            )
        else:
            question_text = raw_content
        question_text = re.sub(r'<audio>', '', question_text, count=1).strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": question_text},
                ]
            }
        ]

        max_retries = 50
        pred = None
        response = None

        for attempt in range(max_retries):
            try:
                if attempt < 2:
                    temperature = 1e-2
                    top_p = 0.1
                    top_k = 1
                else:
                    increase = ((attempt - 2) // 2) * 0.2
                    temperature = 0.1 + increase
                    temperature = min(temperature, 1.0)
                    top_p = 0.9
                    top_k = -1

                response, _ = run_model(
                    model=model,
                    processor=processor,
                    messages=messages,
                    return_audio=return_audio,
                    use_audio_in_video=use_audio_in_video,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

                assert not re.search(r'(.+?)\1{10,}', response.strip()), "LLM repeat continuously"

                pred = extract_choice(response)
                if pred is None:
                    raise ValueError("No A/B/C/D option found in response")
                break
            except Exception as e:
                print(f"[GPU {gpu_id}] Attempt {attempt + 1} Error: {e}")

        result_item = item.copy()
        if pred is not None:
            result_item["response"] = response.strip()
            result_item["pred"] = pred
        else:
            result_item["response"] = response.strip() if response else None
            result_item["pred"] = None
            print(f"[GPU {gpu_id}] Failed for item {idx}")

        # labels = ground truth, directly from the original data
        result_item["labels"] = item["messages"][-1]["content"]
        result_item["logprobs"] = None

        result_queue.put((idx, result_item))

    print(f"[GPU {gpu_id}] Worker finished")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--use_audio_in_video", type=bool, default=True, help="Use audio in video flag")
    parser.add_argument("--return_audio", type=bool, default=False, help="Return audio flag")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    jsonl_path = args.jsonl_path
    save_path = args.save_path
    USE_AUDIO_IN_VIDEO = args.use_audio_in_video
    RETURN_AUDIO = args.return_audio
    NUM_GPUS = args.num_gpus

    # Load data
    jsonl_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            jsonl_data.append(item)

    print(f"Loaded {len(jsonl_data)} items")

    # Create task and result queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for idx, item in enumerate(jsonl_data):
        task_queue.put((idx, item))
    for _ in range(NUM_GPUS):
        task_queue.put(None)

    processes = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(
            target=worker,
            args=(gpu_id, task_queue, result_queue, MODEL_PATH, USE_AUDIO_IN_VIDEO, RETURN_AUDIO)
        )
        p.start()
        processes.append(p)

    # Collect results
    from tqdm import tqdm
    results = {}
    pbar = tqdm(total=len(jsonl_data), desc="Processing")

    finished_count = 0
    while finished_count < len(jsonl_data):
        try:
            idx, result_item = result_queue.get(timeout=60)
            results[idx] = result_item
            finished_count += 1
            pbar.update(1)
        except Exception:
            all_dead = all(not p.is_alive() for p in processes)
            if all_dead:
                print("All workers finished, but some tasks may have failed")
                break

    pbar.close()

    for p in processes:
        p.join()

    # Restore original order
    final_data = [results[i] for i in range(len(jsonl_data)) if i in results]

    # Evaluate accuracy
    correct = 0
    evaluated = 0
    for item in final_data:
        # gt is directly from labels, normalize for comparison
        gt = (item["labels"] or "").strip().upper()
        pred = (item.get("pred") or "").strip().upper()
        is_correct = bool(pred) and pred == gt
        item["correct"] = is_correct
        if pred:
            evaluated += 1
        if is_correct:
            correct += 1

    total = len(final_data)
    acc = correct / total if total > 0 else 0.0

    # Save final results (no gt key, no summary file, no audios path replacement)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Results saved to: {save_path}")
    print(f"Total processed: {len(final_data)}/{len(jsonl_data)}")
    print(f"ACC: {correct}/{total} = {acc:.4f}  (evaluated: {evaluated})")
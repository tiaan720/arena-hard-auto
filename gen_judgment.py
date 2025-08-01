import json
import yaml
import argparse
import os
import concurrent.futures

from tqdm import tqdm

from utils.completion import (
    load_questions,
    registered_api_completion,
    load_questions,
    load_model_answers,
    get_endpoint,
    make_config,
)

from utils.judge_utils import JUDGE_SETTINGS


def get_score(judgment, patterns):
    import re
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def pairwise_judgment(question, baseline, answer, reference, configs, settings):
    prompt_args = {
        "QUESTION": question['prompt'],
        "ANSWER_A": baseline["messages"][-1]["content"]['answer'],
        "ANSWER_B": answer["messages"][-1]["content"]['answer'],
    }
    
    if reference:
        prompt_args[f"REFERENCE"] = reference["messages"][-1]["content"]['answer']
        
    user_prompt = configs["prompt_template"].format(**prompt_args)
    messages = [
        {
            "role": "system", 
            "content": JUDGE_SETTINGS[question["category"]]["system_prompt"],
        },
        {
            "role": "user", 
            "content": user_prompt,
        }
    ]

    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }
    kwargs['temperature'] = configs['temperature']
    kwargs['max_tokens'] = configs['max_tokens']
    
    api_completion_func = registered_api_completion[settings["api_type"]]
    output = api_completion_func(**kwargs)
    
    if output is None:
        return None

    score = get_score(output['answer'], configs["regex_patterns"])

    result = {
        "score": score,
        "judgment": output,
        "prompt": messages,
    }
    return result


def judgment(args):
    answer = args['answer']
    baseline = args['baseline']
    
    output = {
        "uid": args['question']["uid"],
        "category": args['question']["category"],
        "judge": args['configs']['judge_model'],
        "model": answer["model"],
        "baseline": baseline["model"],
        "games": []
    }

    # round 1
    result = pairwise_judgment(
        question=args['question'],
        baseline=baseline,
        answer=answer,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)
        
    # round 2
    result = pairwise_judgment(
        question=args['question'],
        baseline=answer,
        answer=baseline,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)

    with open(args['output_file'], "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/arena-hard-v2.0.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, reference: {configs["reference"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}')

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    if configs["reference"]:
        assert not configs["reference"] in models, "ERROR: one of the models being evaluated is used as reference."
        ref_answers = [answer_dir[model] for model in configs["reference"]]
    else:
        ref_answers = None
    
    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    for model in models:
        # Replace forward slashes with underscores to avoid creating subdirectories
        safe_model_name = model.replace("/", "_")
        output_files[model] = os.path.join(
            output_dir,
            f"{safe_model_name}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    endpoint_settings = endpoint_list[configs["judge_model"]]

    # Collect all tasks first to avoid memory issues with too many futures
    all_tasks = []
    for model in models:
        count = 0
        
        # Skip models that don't have any answers
        if model not in model_answers:
            print(f"Warning: No answers found for model {model}. Skipping.")
            continue
            
        for question in questions:
            uid = question["uid"]

            if uid not in model_answers[model]:
                print(f"Warning: {model} answer to {question['uid']} cannot be found.")
                continue

            if model in existing_judgments and uid in existing_judgments[model]:
                count += 1
                continue

            # Check if baseline exists
            baseline = JUDGE_SETTINGS[question["category"]]["baseline"]
            if baseline not in model_answers or uid not in model_answers[baseline]:
                print(f"Warning: Baseline {baseline} answer to {question['uid']} cannot be found.")
                continue

            kwargs = {}
            kwargs["question"] = question
            kwargs["answer"] = model_answers[model][uid]
            kwargs["baseline"] = model_answers[baseline][uid]
            
            if ref_answers:
                kwargs["reference"] = [ref_answer[uid] for ref_answer in ref_answers]
            else:
                kwargs["reference"] = None
                
            kwargs["configs"] = configs
            kwargs["settings"] = endpoint_settings
            kwargs["output_file"] = output_files[model]
            
            all_tasks.append(kwargs)

        if count > 0:
            print(f"{count} number of existing judgments")
    
    print(f"Total tasks to process: {len(all_tasks)}")
    
    # Process tasks in batches to avoid memory issues
    batch_size = 100  # Process 100 tasks at a time
    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_settings["parallel"]) as executor:
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]
            futures = []
            
            for task in batch:
                future = executor.submit(judgment, task)
                futures.append(future)
            
            print(f"Processing batch {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size}")
            for future in tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(futures),
                desc=f"Batch {i//batch_size + 1}"
            ):
                future.result()

import argparse
import json
from tqdm import tqdm
import datasets
import transformers

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, help="checkpoint, like `THUDM/chatglm-6b`")
parser.add_argument("--input_file", type=str, help="Instruction for path of data file，each line in the file is in json format and contains one input and one output")
parser.add_argument("--prompt_key", type=str, default=f"prompt", help="Input field of instruction: prompt")
parser.add_argument("--target_key", type=str, default=f"target", help="Output field of instruction: target")
parser.add_argument("--save_name", type=str, default=f"temp", help="The location of the tokenized data set")
parser.add_argument("--max_seq_length", type=int, default=2040)
parser.add_argument("--skip_overlength", type=bool, default=False)
args = parser.parse_args()
model_checkpoint = args.model_checkpoint
# base_model_name = model_checkpoint.split('/')[-1]


# model_checkpoint = "THUDM/chatglm-6b"
# model_checkpoint = "baichuan-inc/baichuan-7B"


################ Preprocessing ################
# Define preprocess function to concatenate token IDs
def preprocess(tokenizer, config, example, max_seq_length, prompt_key, target_key):
    prompt = example[prompt_key]
    target = example[target_key]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target, max_length=max_seq_length, truncation=True, add_special_tokens=False)
    # The input and output of instruction are combined, and the classic causal-LM next word prediction method
    # is used for training

    # Concatenate these IDs into a single sequence which is meant to be an input for a causal language model
    input_ids = prompt_ids + target_ids + [config.eos_token_id] # eos: the end of a sentence
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}



################ Loading jsonl file ################
def read_jsonl(path, max_seq_length, prompt_key,target_key,skip_overlength=False):
    # Tokenizer and configuration loading
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_checkpoint, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_checkpoint, trust_remote_code=True, device_map='auto')
    # Load jsonl file and read data line by line
    with open(path, "r") as f:
        for line in tqdm(f.readlines()): # tqdm shows the progress bar for each iteration
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length,prompt_key,target_key)
            # Skip an example and continue with next line if skip_overlength is true and
            # the length of the input_ids is greater than max sequence length
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


# Input files are placed in the data folder
# Output files are placed in the data/tokenized_data folder
input_file_path = f'data/{args.input_file}'
save_path = f"data/tokenized_data/{args.save_name}"
dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(input_file_path, args.max_seq_length, args.prompt_key,args.target_key,args.skip_overlength)
)

dataset.save_to_disk(save_path)

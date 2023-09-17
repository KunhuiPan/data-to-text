from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os
from pprint import pprint as print

model_path = "/home/user/imported_models/chatglm-6b-20230419"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


@dataclass
class FinetuneArguments:
    tokenized_dataset: str = field(default=" ")  # Dataset folder after tokenization
    model_path: str = field(default=" ")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]  # prompt length
        labels = (
                [-100] * (seq_len - 1) + ids[(seq_len - 1):] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        # Since the model provided to the trainer is actually of type PeftModel, save_pretrained here will directly
        # use the save method of PeftModel. Therefore, only the LoRA weights will be saved
        self.model.save_pretrained(output_dir)
        # from transformers.trainer import TRAINING_ARGS_NAME
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # saved_params = {
        #     k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        # }
        # torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    writer = SummaryWriter()
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # load dataset
    dataset = datasets.load_from_disk('data/tokenized_data' + finetune_args.tokenized_dataset)
    print(f"\n{len(dataset)=}\n")

    # init model
    model = AutoModel.from_pretrained(
        model_path, load_in_8bit=False, trust_remote_code=True,
        device_map="auto"  # Different layers of the model are automatically assigned to different GPUs for computation
        # device_map={'':torch.cuda.current_device()}
    )
    print(model.hf_device_map)

    model.gradient_checkpointing_enable()
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    model.enable_input_require_grads()
    # Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
    # the model weights fixed. model.is_parallelizable = True A flag indicating whether this model supports model
    # parallelization. When set to True, model parallelization may be initiated and data parallelism is turned off,
    # allowing a model to be chunked across multiple GPUs model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    # model.config.use_cache = (
    #     False
    # )

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)

    # start train
    model.save_pretrained(training_args.output_dir)
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

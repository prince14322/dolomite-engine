import json
import logging
from argparse import ArgumentParser
from copy import deepcopy
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import torch
import transformers
from peft import PromptTuningInit
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .enums import ArgsFileExtension, AttentionImplementation, DistributedBackend, Mode, PaddingSide, TuningMethod
from .utils import get_world_size, load_yaml, log_rank_0, run_rank_n, set_logger


_ARGS_FILE_EXTENSION: ArgsFileExtension = None


def _check_not_None(object_name_list: List[Tuple[Any, str]]) -> None:
    for obj, name in object_name_list:
        assert obj is not None, f"{name} cannot be None"


class BaseArgs(BaseModel):
    model_config = ConfigDict(extra="allow")

    def to_dict(self) -> dict:
        copied = deepcopy(self)

        for key, value in copied:
            if isinstance(value, BaseArgs):
                result = value.to_dict()
            elif isinstance(value, list):
                result = []
                for v in value:
                    if isinstance(v, BaseArgs):
                        result.append(v.to_dict())
            elif isinstance(value, Enum):
                result = value.value
            elif isinstance(value, type):
                result = value.__name__
            else:
                result = value

            setattr(copied, key, result)

        return vars(copied)


class RandomArgs(BaseArgs):
    # random seed
    seed: int = 42


class TokenizerArgs(BaseArgs):
    # add special tokens to the tokenizer
    additional_special_tokens: Optional[List[str]] = None
    # padding side
    padding_side: Optional[PaddingSide] = None


class ModelArgs(BaseArgs):
    # model name on huggingface hub
    model_name: str = None
    # model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM
    model_class: str = None
    # dtype to use for training / inference
    dtype: str = "float32"
    # trust remote code for models that are not directly supported by HuggingFace yet
    trust_remote_code: bool = False
    # attention implementation (only works with GPTMegatronForCausalLM)
    attention_implementation: Optional[AttentionImplementation] = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.model_name, "model_name"), (self.model_class, "model_class")])

        # model_class
        if self.attention_implementation is None:
            assert self.model_class in [
                AutoModelForCausalLM.__name__,
                AutoModelForSeq2SeqLM.__name__,
            ], f"attention implementation is not supported with {AutoModelForCausalLM.__name__} or {AutoModelForSeq2SeqLM.__name__}"

            self.model_class: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM] = getattr(
                transformers, self.model_class
            )
        else:
            try:
                from ibm_models import GPTMegatronForCausalLM

                self.model_class = GPTMegatronForCausalLM
            except ImportError:
                raise ImportError(
                    "please pip install ibm-models: https://github.ibm.com/ai-models-architectures/ibm-models"
                )

        # dtype
        self.dtype = getattr(torch, self.dtype)
        assert self.dtype in [torch.float32, torch.float16, torch.bfloat16], f"unexpected dtype '{self.dtype}'"

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["dtype"] = str(self.dtype).split(".")[1]
        return result


class PromptTuningArgs(BaseArgs):
    # prompt tuning init method
    prompt_tuning_init: PromptTuningInit = None
    # prompt tuning init text
    prompt_tuning_init_text: Optional[str] = None
    # number of virtual tokens for PEFT
    num_virtual_tokens: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.prompt_tuning_init, "prompt_tuning_init")])

        if self.prompt_tuning_init == PromptTuningInit.RANDOM:
            assert (
                self.prompt_tuning_init_text is None
            ), f"prompt_tuning_init_text '{self.prompt_tuning_init_text}' was specified with RANDOM init method"
        elif self.prompt_tuning_init == PromptTuningInit.TEXT:
            assert (
                self.prompt_tuning_init_text is not None
            ), f"prompt_tuning_init_text needs to be specified with TEXT init method"


class LoRAArgs(BaseArgs):
    # lora rank
    lora_rank: int = None
    # the scaling factor for the low-rank matrices
    lora_alpha: float = 32.0
    # the dropout probability of the LoRA layers
    lora_dropout: float = 0.1

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.lora_rank, "lora_rank")])


class TuningArgs(BaseArgs):
    # type of tuning, full finetuning or PEFT
    tuning_method: TuningMethod = None
    # prompt tuning related arguments
    prompt_tuning_args: Optional[PromptTuningArgs] = None
    # lora related arguments
    lora_args: Optional[LoRAArgs] = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.tuning_method, "tuning_method")])

        # check whether the arguments specified are valid
        if self.tuning_method == TuningMethod.full_finetuning:
            assert self.prompt_tuning_args is None, "prompt_tuning_args should not be specified with full_finetuning"
            assert self.lora_args is None, "lora_args should not be specified with full_finetuning"
        elif self.tuning_method == TuningMethod.prompt_tuning:
            assert self.lora_args is None, "lora_args should not be specified with promt_tuning"
        elif self.tuning_method == TuningMethod.lora:
            assert self.prompt_tuning_args is None, "prompt_tuning_args should not be specified with lora"


class TrainingParameters(BaseArgs):
    # whether to use sequential sampler for validation
    ignore_sampling_proportion_for_validation: bool = False
    # number of training steps
    num_training_steps: Optional[int] = None
    # gradient accumulation steps
    gradient_accumulation_steps: int = 1
    # interval for evaluation
    eval_interval: Optional[int] = None
    # batch size per GPU for ZeRO-DP
    batch_size_per_gpu: int = None
    # whether to use val dataset for validation during training
    eval_during_training: bool = True

    def model_post_init(self, __context: Any) -> None:
        _check_not_None(
            [(self.num_training_steps, "num_training_steps"), (self.batch_size_per_gpu, "batch_size_per_gpu")]
        )

        # eval_interval
        if self.eval_during_training:
            _check_not_None([(self.eval_interval, "eval_interval")])


class SaveArgs(BaseArgs):
    # path to save checkpoints
    save_path: str = None
    # interval for checkpointing
    save_interval: int = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.save_path, "save_path"), (self.save_interval, "save_interval")])


class LoadArgs(BaseArgs):
    # path to load checkpoints
    load_path: str = None
    # iteration to load
    iteration: int = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.load_path, "load_path"), (self.iteration, "iteration")])


class DatasetArgs(BaseArgs):
    # optimizer class
    class_name: str = None
    # class args for optimizer
    class_args: dict = {}
    # dataset name
    data_name: str = None
    # formatting to use for input
    input_format: str = "__input__"
    # formatting to use for output
    output_format: str = "__output__"
    # data sampling proportions
    data_sampling_ratio: int = None
    # max tokens for input text
    max_input_tokens: Optional[int] = None
    # max tokens for output text
    max_output_tokens: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None(
            [
                (self.class_name, "dataset class_name"),
                (self.data_sampling_ratio, "data_sampling_ratio"),
                (self.data_name, "data_name"),
            ]
        )

        # data_sampling_ratios
        assert self.data_sampling_ratio > 0, "data_sampling_ratio should be a positive integer"


class OptimizerArgs(BaseArgs):
    # optimizer class
    class_name: str = "ApexFusedAdam"
    # class args for optimizer
    class_args: dict = {
        "lr": 1e-5,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1e-10,
    }

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.class_name, "optimizer class_name")])


class LRSchedulerArgs(BaseArgs):
    # learning rate schedule
    lr_schedule: str = "cosine"
    # warmup steps
    num_warmup_steps: int = 200


class DistributedArgs(BaseArgs):
    # ZeRO stage
    stage: int = 3
    # distributed backend to use
    distributed_backend: DistributedBackend = DistributedBackend.torch
    # overlap communication with computation
    overlap_comm: bool = False
    # use contiguous buffers for gradients, requires more memory if enabled
    contiguous_gradients: bool = False
    # train with CPU offloading to save GPU memory
    cpu_offload: bool = False
    # whether to use gradient checkpointing, enabling leads to lower memory usage with increased step time
    gradient_checkpointing: bool = False


class LoggingArgs(BaseArgs):
    # logging level
    logging_level: str = "INFO"
    # log interval
    log_interval: int = 1
    # aim repo, experiment logs are saved here
    aim_repo: Optional[str] = None
    # name of the experiment
    experiment_name: Optional[str] = None


class TrainingArgs(BaseArgs):
    # randomization related arguments
    random_args: RandomArgs = RandomArgs()
    # tokenizer related arguments
    tokenizer_args: TokenizerArgs = TokenizerArgs()
    # model related arguments
    model_args: ModelArgs = None
    # tuning related arguments
    tuning_args: TuningArgs = None
    # optimizer related arguments
    optimizer_args: OptimizerArgs = OptimizerArgs()
    # lr_scheduler related arguments
    lr_scheduler_args: LRSchedulerArgs = LRSchedulerArgs()
    # list of datasets to use
    datasets: List[DatasetArgs] = []
    # save related arguments
    save_args: SaveArgs = None
    # load related arguments
    load_args: Optional[LoadArgs] = None
    # training parameters
    training_parameters: Optional[TrainingParameters] = None
    # logging related arguments
    logging_args: LoggingArgs = LoggingArgs()
    # distributed training related arguments
    distributed_args: DistributedArgs = DistributedArgs()

    def model_post_init(self, __context: Any) -> None:
        _check_not_None(
            [
                (self.model_args, "model_args"),
                (self.tuning_args, "tuning_args"),
                (self.save_args, "save_args"),
                (self.datasets, "datasets"),
            ]
        )

        # datasets
        _check_datasets(self.datasets)


class GenerationParameters(BaseArgs):
    # batch size
    batch_size: int = None
    # sample or greedy
    do_sample: Optional[bool] = None
    # max new tokens to generate
    max_new_tokens: int = None
    # temperature
    temperature: Optional[float] = None
    # top k
    top_k: Optional[int] = None
    # top p
    top_p: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.batch_size, "batch_size"), (self.max_new_tokens, "max_new_tokens")])


class InferenceArgs(BaseArgs):
    # randomization related arguments
    random_args: RandomArgs = RandomArgs()
    # tokenizer related arguments
    tokenizer_args: TokenizerArgs = TokenizerArgs()
    # model related arguments
    model_args: ModelArgs = None
    # tuning related arguments
    tuning_args: TuningArgs = None
    # list of datasets to use
    datasets: List[DatasetArgs] = []
    # load related arguments
    load_args: Optional[LoadArgs] = None
    # generation parameters
    generation_parameters: GenerationParameters = None
    # logging related arguments
    logging_args: LoggingArgs = LoggingArgs()
    # output dir
    output_dir: str = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None(
            [
                (self.model_args, "model_args"),
                (self.tuning_args, "tuning_args"),
                (self.datasets, "datasets"),
                (self.generation_parameters, "generation_parameters"),
                (self.output_dir, "output_dir"),
            ]
        )

        # datasets
        _check_datasets(self.datasets)


class ExportArgs(BaseArgs):
    # tokenizer related arguments
    tokenizer_args: TokenizerArgs = TokenizerArgs()
    # model related arguments
    model_args: ModelArgs = None
    # tuning related arguments
    tuning_args: TuningArgs = None
    # load related arguments
    load_args: LoadArgs = None
    # logging related arguments
    logging_args: LoggingArgs = LoggingArgs()
    # export path
    export_path: str = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None(
            [
                (self.model_args, "model_args"),
                (self.tuning_args, "tuning_args"),
                (self.load_args, "load_args"),
                (self.export_path, "export_path"),
            ]
        )


_MODE_ARGS_MAP = {
    Mode.training: TrainingArgs,
    Mode.inference: InferenceArgs,
    Mode.export: ExportArgs,
}


def get_args(mode: Mode) -> Union[TrainingArgs, InferenceArgs, ExportArgs]:
    """get args for training / inference

    Args:
        mode (Mode): training / inference mode for running the program

    Returns:
        Union[TrainingArgs, InferenceArgs, ExportArgs]: args for training / inference
    """

    global _ARGS_FILE_EXTENSION

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path for the config")
    args = parser.parse_args()

    if args.config.endswith("json"):
        config: dict = json.load(open(args.config, "r"))
        _ARGS_FILE_EXTENSION = ArgsFileExtension.json
    elif args.config.endswith("yaml") or args.config.endswith("yml"):
        config: dict = load_yaml(args.config)
        _ARGS_FILE_EXTENSION = ArgsFileExtension.yaml

    args: Union[TrainingArgs, InferenceArgs, ExportArgs] = _MODE_ARGS_MAP[mode](**config)

    set_logger(args.logging_args.logging_level)
    log_args(args)

    return args


@run_rank_n
def log_args(args: Union[TrainingArgs, InferenceArgs, ExportArgs]) -> None:
    """log args

    Args:
        args (Union[TrainingArgs, InferenceArgs, ExportArgs]): args for training / inference
    """

    def _iterate_args_recursively(
        args: Union[TrainingArgs, InferenceArgs, ExportArgs, dict, BaseArgs], prefix: str = ""
    ) -> None:
        result = []

        if isinstance(args, BaseArgs):
            args = vars(args)

        p = len(prefix)

        for k, v in args.items():
            suffix = "." * (48 - len(k) - p)

            if isinstance(v, (BaseArgs, dict)):
                if isinstance(v, dict) and len(v) == 0:
                    result.append(f"{prefix}{k} {suffix} " + r"{}")
                else:
                    kv_list_subargs = _iterate_args_recursively(v, prefix + " " * 4)
                    result.append(f"{prefix}{k}:\n" + "\n".join(kv_list_subargs))
            elif isinstance(v, list) and all([isinstance(v_, (BaseArgs, dict)) for v_ in v]):
                kv_list_subargs = []
                for v_ in v:
                    v_ = _iterate_args_recursively(v_, prefix + " " * 4)
                    kv_list_subargs.append(f"\n".join(v_))
                result.append(f"{prefix}{k}:\n" + ("\n" + " " * (p + 4) + "*" * (44 - p) + "\n").join(kv_list_subargs))
            else:
                result.append(f"{prefix}{k} {suffix} " + str(v))

        result.sort(key=lambda x: x.lower())
        return result

    log_rank_0(logging.INFO, f"total GPUs = {get_world_size()}")
    log_rank_0(logging.INFO, "------------------------ arguments ------------------------")
    for line in _iterate_args_recursively(args):
        line = line.split("\n")
        for l in line:
            log_rank_0(logging.INFO, l)
    log_rank_0(logging.INFO, "-------------------- end of arguments ---------------------")


def get_args_file_extension() -> ArgsFileExtension:
    assert _ARGS_FILE_EXTENSION is not None, "args file extesnion is not set"
    return _ARGS_FILE_EXTENSION


def _check_datasets(datasets: List[DatasetArgs]) -> None:
    assert len(datasets) != 0, "datasets cannot be an empty list"
    # check data_names are unique
    assert len(datasets) == len(
        set([dataset.data_name for dataset in datasets])
    ), "data_name should be unique for each dataset"
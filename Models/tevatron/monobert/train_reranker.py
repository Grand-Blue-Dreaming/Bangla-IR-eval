import logging
import os
import sys
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments

from tevatron.reranker.modeling import RerankerModel
from tevatron.reranker.data import RerankerTrainDataset, RerankerTrainCollator, \
    HFRerankDataset, RerankerInferenceDataset, RerankerInferenceCollator

from tevatron.reranker.trainer import RerankerTrainer
from tevatron.datasets import HFTrainDataset

from evaluation_callback import EvaluationCallback
from evaluation_arguments import EvaluationArguments


logger = logging.getLogger(__name__)


def read_result(path):
    retrieval_results = {}
    with open(path) as f:
        for line in f:
            if len(line.rstrip().split()) == 3:
                qid, pid, score = line.rstrip().split()
            else:
                qid, _, pid, _, score, _ = line.rstrip().split()
            if qid not in retrieval_results:
                retrieval_results[qid] = []
            retrieval_results[qid].append((pid, float(score)))
    return retrieval_results

def prepare_rerank_input(retrieval_results, topics_path, collection_path, depth=1000):
    retrieval_results = retrieval_results
    output_path = "rerank_input_file.bm25.jsonl"

    query_id_map = {}
    with open(topics_path, 'r') as ft:
        topics = ft.readlines()
        for topic in topics:
            query_id = topic.split('\t')[0]
            query = topic.split('\t')[1].strip()
            query_id_map[query_id] = query

    corpus_id_map = {}
    with open(collection_path, 'r') as fc:
        for line in fc:
            doc = json.loads(line)
            doc['docid'] = doc.pop('id')
            doc['text'] = doc.pop('contents')
            corpus_id_map[doc['docid']] = doc

    retrieval_results = read_result(retrieval_results)

    with open(output_path, 'w') as f:
        for qid in retrieval_results:
            query = query_id_map[qid]
            pid_and_scores = retrieval_results[qid]
            for item in pid_and_scores[:depth]:
                pid, score = item
                psg_info = corpus_id_map[pid]
                psg_info['score'] = score
                psg_info['query_id'] = qid
                psg_info['query'] = query
                f.write(json.dumps(psg_info)+'\n')



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, EvaluationArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        eval_args: EvaluationArguments

    eval_args.fp16 = training_args.fp16

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    model = RerankerModel.build(
        model_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = RerankerTrainDataset(data_args, train_dataset.process(), tokenizer)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()


    # evaluation data preparation
    # dataset
    prepare_rerank_input(eval_args.retrieval_results, eval_args.topics_path, eval_args.collection_path)

    eval_data_args = DataArguments(
        dataset_name='json',
        encode_in_path='rerank_input_file.bm25.jsonl',
        encoded_save_path=eval_args.output_save_path,
    )    

    rerank_dataset = HFRerankDataset(tokenizer=tokenizer, data_args=eval_data_args, cache_dir=None)
    rerank_dataset = RerankerInferenceDataset(
        rerank_dataset.process(),
        tokenizer, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len
    )

    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=RerankerInferenceCollator(
            tokenizer,
            max_length=data_args.q_max_len+data_args.p_max_len,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RerankerTrainCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        callbacks=[EvaluationCallback(rerank_loader, eval_args)],
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

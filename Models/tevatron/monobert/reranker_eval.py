import json
from tqdm import tqdm

import logging
import os
import pickle
import sys
from contextlib import nullcontext
from unittest import result

import numpy as np

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from tevatron.reranker.data import HFRerankDataset, RerankerInferenceDataset, RerankerInferenceCollator
from tevatron.reranker.modeling import RerankerModel

from tevatron.arguments import DataArguments

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--collection_path', type=str, required=True)
parser.add_argument('--topics_path', type=str, required=True)
parser.add_argument('--qrels_path', type=str, required=True)
parser.add_argument('--retrieval_results', type=str, required=True)
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--output_save_path', type=str, default="run.monobert.txt", required=False)
parser.add_argument('--depth', type=int, default=1000, required=False)
parser.add_argument('--cache_dir', type=str, default=None, required=False)
parser.add_argument('--fp16', type=bool, default=True, required=False)
parser.add_argument('--per_device_eval_batch_size', type=int, default=32, required=False)
parser.add_argument('--dataloader_num_workers', type=int, default=12, required=False)


args = parser.parse_args()


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

def prepare_rerank_input():
    retrieval_results = args.retrieval_results
    output_path = "rerank_input_file.bm25.jsonl"

    query_id_map = {}
    with open(args.topics_path, 'r') as ft:
        topics = ft.readlines()
        for topic in tqdm(topics):
            query_id = topic.split('\t')[0]
            query = topic.split('\t')[1].strip()
            query_id_map[query_id] = query

    corpus_id_map = {}
    with open(args.collection_path, 'r') as fc:
        for line in fc:
            doc = json.loads(line)
            doc['docid'] = doc.pop('id')
            doc['text'] = doc.pop('contents')
            corpus_id_map[doc['docid']] = doc

    retrieval_results = read_result(retrieval_results)

    with open(output_path, 'w') as f:
        for qid in tqdm(retrieval_results):
            query = query_id_map[qid]
            pid_and_scores = retrieval_results[qid]
            for item in pid_and_scores[:args.depth]:
                pid, score = item
                psg_info = corpus_id_map[pid]
                psg_info['score'] = score
                psg_info['query_id'] = qid
                psg_info['query'] = query
                f.write(json.dumps(psg_info)+'\n')

prepare_rerank_input()


# model
num_labels = 1
# dataset
data_args = DataArguments(
    dataset_name='json',
    encode_in_path='rerank_input_file.bm25.jsonl',
    q_max_len=64,p_max_len=256,
    encoded_save_path=args.output_save_path,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    num_labels=num_labels,
    cache_dir=args.cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
)

model = RerankerModel.load(
    model_name_or_path=args.model_name_or_path,
    config=config,
    cache_dir=args.cache_dir,
)

rerank_dataset = HFRerankDataset(tokenizer=tokenizer, data_args=data_args, cache_dir=args.cache_dir)
rerank_dataset = RerankerInferenceDataset(
    rerank_dataset.process(),
    tokenizer, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len
)

rerank_loader = DataLoader(
    rerank_dataset,
    batch_size=args.per_device_eval_batch_size,
    collate_fn=RerankerInferenceCollator(
        tokenizer,
        max_length=data_args.q_max_len+data_args.p_max_len,
        padding='max_length'
    ),
    shuffle=False,
    drop_last=False,
    num_workers=args.dataloader_num_workers,
)
model = model.to(device)
model.eval()
all_results = {}

for (batch_query_ids, batch_text_ids, batch) in tqdm(rerank_loader):
    with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
        with torch.no_grad():
            for k, v in batch.items():
                batch[k] = v.to(device)
            model_output = model(batch)
            scores = model_output.scores.cpu().detach().numpy()
            for i in range(len(scores)):
                qid = batch_query_ids[i]
                docid = batch_text_ids[i]
                score = scores[i][0]
                if qid not in all_results:
                    all_results[qid] = []
                all_results[qid].append((docid, score))

with open(data_args.encoded_save_path, 'w') as f:
    for qid in all_results:
        results = sorted(all_results[qid], key=lambda x: x[1], reverse=True)
        for docid, score in results:
            f.write(f'{qid}\t{docid}\t{score}\n')

res = os.popen(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 -m recall.100 {args.qrels_path} {data_args.encoded_save_path}").read()
print(res)

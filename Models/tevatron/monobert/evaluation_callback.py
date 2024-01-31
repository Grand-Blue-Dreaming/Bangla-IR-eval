import logging
import os

import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback, AutoConfig

from contextlib import nullcontext

from tevatron.reranker.modeling import RerankerModel

from evaluation_arguments import EvaluationArguments

logger = logging.getLogger(__name__)

class EvaluationCallback(TrainerCallback):

    def __init__(self, rerank_loader: DataLoader, eval_args: EvaluationArguments) -> None:
        super().__init__()
        self.rerank_loader = rerank_loader
        self.eval_args = eval_args

    def on_epoch_end(self, args, state, control, **kwargs):

        # save epoch checkpoint
        checkpoint_save_path = f"monobert-reranker-epoch-{state.epoch}"
        model = kwargs.get('model', None)
        model.save(checkpoint_save_path)

        # model
        config = AutoConfig.from_pretrained(
            checkpoint_save_path,
            num_labels=1,
        )
        model = RerankerModel.load(
            model_name_or_path=checkpoint_save_path,
            config=config,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        model = model.to(device)
        model.eval()
        all_results = {}

        for (batch_query_ids, batch_text_ids, batch) in self.rerank_loader:
            with torch.cuda.amp.autocast() if self.eval_args.fp16 else nullcontext():
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

        with open(self.eval_args.output_save_path, 'w') as f:
            for qid in all_results:
                results = sorted(all_results[qid], key=lambda x: x[1], reverse=True)
                for docid, score in results:
                    f.write(f'{qid}\t{docid}\t{score}\n')

        res = os.popen(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 -m recall.100 {self.eval_args.qrels_path} {self.eval_args.output_save_path}").read()
        logger.info(f"Epoch {state.epoch}, Dataset: {self.eval_args.eval_dataset_name}\n {res[-78:]}")
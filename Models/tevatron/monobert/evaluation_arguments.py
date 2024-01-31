from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvaluationArguments:
    eval_dataset_name: str = field(
        default=None, metadata={"help": "dataset name"}
    )
    collection_path: str = field(
        default=None, metadata={"help": "path to the collection.jsonl file containing the evaluation passage collection"}
    )
    topics_path: str = field(
        default=None, metadata={"help": "path to the topics.tsv file containing the evaluation queries"}
    )
    qrels_path: str = field(
        default=None, metadata={"help": "path to the qrels.txt file containing the evaluation relevance judgements"}
    )
    retrieval_results: str = field(
        default=None, metadata={"help": "path to the initial retrieval run file, e.g. run.bm25.data.txt"}
    )
    output_save_path: Optional[str] = field(
        default=f"run.monobert.txt", metadata={"help": "path to save the evaluation run file"}
    )



import argparse
import json
import random

import numpy as np
import torch

from blanc import BlancHelp, BlancTune
from blanc.utils import Defaults


def main():
    parser = argparse.ArgumentParser(
        prog='blanc', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        'type', type=str, choices=['help', 'tune'], help='BLANC-help or BLANC-tune'
    )

    input_parser = parser.add_argument_group('input arguments')
    input_parser.add_argument('--doc', type=str, help='single input document')
    input_parser.add_argument('--summary', type=str, help='single input summary')
    input_parser.add_argument(
        '--single_json',
        type=str,
        help='filename for single document summary pair',
        metavar='FILENAME',
    )
    input_parser.add_argument(
        '--pairs_json',
        type=str,
        help='filename for list of document summary pairs',
        metavar='FILENAME',
    )
    input_parser.add_argument(
        '--doc_summaries_json',
        type=str,
        help='filename for list of documents, each with a list of summaries',
        metavar='FILENAME',
    )
    input_parser.add_argument(
        '--doc_key',
        type=str,
        help='json key for the input document',
        metavar='KEY',
        default=Defaults.doc_key,
    )
    input_parser.add_argument(
        '--summary_key',
        type=str,
        help='json key for the input summary (single_json or pairs_json input)',
        metavar='KEY',
        default=Defaults.summary_key,
    )
    input_parser.add_argument(
        '--summaries_key',
        type=str,
        help='json key for the input summaries (doc_summaries_json input)',
        metavar='KEY',
        default=Defaults.summaries_key,
    )
    input_parser.add_argument(
        '--output_json',
        type=str,
        help='filename for output file, or None to print to STDOUT',
        metavar='FILENAME',
    )

    blanc_parser = parser.add_argument_group('arguments for BLANC-help and BLANC-tune')
    blanc_parser.add_argument(
        '--model_name',
        type=str,
        choices=['bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased'],
        help='BERT model type',
        default=Defaults.model_name,
        metavar='NAME',
    )
    blanc_parser.add_argument(
        '--measure',
        type=str,
        choices=['improve', 'relative'],
        help='measure improve or relative, as defined in the paper',
        default=Defaults.measure,
    )
    blanc_parser.add_argument(
        '--gap',
        type=int,
        help='distance between words to mask during inference',
        default=Defaults.gap,
    )
    blanc_parser.add_argument(
        '--gap_mask',
        type=int,
        help='number of tokens to mask at each designated position during inference',
        default=Defaults.gap_mask,
    )
    blanc_parser.add_argument(
        '--gap_tune',
        type=int,
        help='distance between words to mask during finetuning',
        default=Defaults.gap,
    )
    blanc_parser.add_argument(
        '--gap_mask_tune',
        type=int,
        help='number of tokens to mask at each designated position during finetuning',
        default=Defaults.gap_mask,
    )
    blanc_parser.add_argument(
        '--min_token_length_normal',
        type=int,
        help=(
            'minimum number of chars in normal tokens to mask, where a normal token is '
            'a whole word'
        ),
        default=Defaults.min_token_length_normal,
        metavar='LEN',
    )
    blanc_parser.add_argument(
        '--min_token_length_lead',
        type=int,
        help='minimum number of chars in lead token to mask, where a lead token begins a word',
        default=Defaults.min_token_length_lead,
        metavar='LEN',
    )
    blanc_parser.add_argument(
        '--min_token_length_followup',
        type=int,
        help=(
            'minimum number of chars in followup token to mask, where a followup token '
            'continues a word'
        ),
        default=Defaults.min_token_length_followup,
        metavar='LEN',
    )
    blanc_parser.add_argument(
        '--min_token_length_normal_tune',
        type=int,
        help=(
            'minimum number of chars in normal tokens to mask at tuning, where a normal token is '
            'a whole word'
        ),
        default=Defaults.min_token_length_normal_tune,
        metavar='LEN',
    )
    blanc_parser.add_argument(
        '--min_token_length_lead_tune',
        type=int,
        help='minimum number of chars in lead token to mask at tuning, where a lead token begins a word',
        default=Defaults.min_token_length_lead_tune,
        metavar='LEN',
    )
    blanc_parser.add_argument(
        '--min_token_length_followup_tune',
        type=int,
        help=(
            'minimum number of chars in followup token to mask at tuning, where a followup token '
            'continues a word'
        ),
        default=Defaults.min_token_length_followup_tune,
        metavar='LEN',
    )
    blanc_parser.add_argument(
        '--device', type=str, help='cpu or cuda device', default=Defaults.device,
    )
    blanc_parser.add_argument(
        '--random_seed',
        type=int,
        help='random seed for python and torch',
        default=Defaults.random_seed,
        metavar='SEED',
    )
    blanc_parser.add_argument(
        '--inference_batch_size',
        type=int,
        help='batch size to use during inference',
        default=Defaults.inference_batch_size,
        metavar='SIZE',
    )
    blanc_parser.add_argument(
        '--inference_mask_evenly',
        type=bool,
        help=(
            'when True, mask every `gap` tokens (`gap_mask` tokens at once) that are longer than `min_token_length`'
            'during finetuning, when False randomly mask tokens with probability 0.15'
        ),
        default=Defaults.inference_mask_evenly,
        metavar='MASK_EVENLY',
    )

    help_parser = parser.add_argument_group('BLANC-help arguments')
    help_parser.add_argument(
        '--filler_token',
        type=str,
        help='token to use as filler in lieu of summary',
        default=Defaults.filler_token,
        metavar='TOKEN',
    )
    help_parser.add_argument(
        '--help_sep',
        type=str,
        help=(
            "token to use to separate the summary or filler from the sentence, "
            "or '' for no separator"
        ),
        default=Defaults.help_sep,
        metavar='SEP',
    )

    tune_parser = parser.add_argument_group('BLANC-tune arguments')
    tune_parser.add_argument(
        '--finetune_batch_size',
        type=int,
        help='batch size to use when finetuning on summary',
        default=Defaults.finetune_batch_size,
        metavar='SIZE',
    )
    tune_parser.add_argument(
        '--finetune_epochs',
        type=int,
        help='number of epochs to train for when finetuning on summary',
        default=Defaults.finetune_epochs,
        metavar='EPOCHS',
    )
    tune_parser.add_argument(
        '--finetune_mask_evenly',
        type=bool,
        help=(
            'when True, mask every `gap` tokens (`gap_mask` tokens at once) that are longer than `min_token_length`'
            'during finetuning, when False randomly mask tokens with probability 0.15'
        ),
        default=Defaults.finetune_mask_evenly,
        metavar='MASK_EVENLY',
    )
    tune_parser.add_argument(
        '--finetune_chunk_size',
        type=int,
        help='number of summary tokens to use at a time when finetuning',
        default=Defaults.finetune_chunk_size,
        metavar='SIZE',
    )
    tune_parser.add_argument(
        '--finetune_chunk_stride',
        type=int,
        help='number of tokens between summary chunks for finetuning',
        default=Defaults.finetune_chunk_stride,
        metavar='STRIDE',
    )
    tune_parser.add_argument(
        '--learning_rate',
        type=float,
        help='learning rate when finetuning on summary',
        default=Defaults.learning_rate,
        metavar='LR',
    )
    tune_parser.add_argument(
        '--warmup_steps',
        type=int,
        help='warmup steps when finetuning on summary',
        default=Defaults.warmup_steps,
        metavar='STEPS',
    )

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.type == 'help':
        model = BlancHelp(
            model_name=args.model_name,
            measure=args.measure,
            gap=args.gap,
            gap_mask=args.gap_mask,
            gap_tune=args.gap_tune,
            gap_mask_tune=args.gap_mask_tune,
            min_token_length_normal=args.min_token_length_normal,
            min_token_length_lead=args.min_token_length_lead,
            min_token_length_followup=args.min_token_length_followup,
            min_token_length_normal_tune=min_token_length_normal_tune,
            min_token_length_lead_tune=min_token_length_lead_tune,
            min_token_length_followup_tune=min_token_length_followup_tune,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
            inference_mask_evenly=args.inference_mask_evenly,
            filler_token=args.filler_token,
            help_sep=args.help_sep,
        )
    elif args.type == 'tune':
        model = BlancTune(
            model_name=args.model_name,
            measure=args.measure,
            gap=args.gap,
            gap_mask=args.gap_mask,
            gap_tune=args.gap_tune,
            gap_mask_tune=args.gap_mask_tune,
            min_token_length_normal=args.min_token_length_normal,
            min_token_length_lead=args.min_token_length_lead,
            min_token_length_followup=args.min_token_length_followup,
            min_token_length_normal_tune=min_token_length_normal_tune,
            min_token_length_lead_tune=min_token_length_lead_tune,
            min_token_length_followup_tune=min_token_length_followup_tune,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
            inference_mask_evenly=args.inference_mask_evenly,
            finetune_batch_size=args.finetune_batch_size,
            finetune_epochs=args.finetune_epochs,
            finetune_mask_evenly=args.finetune_mask_evenly,
            finetune_chunk_size=args.finetune_chunk_size,
            finetune_chunk_stride=args.finetune_chunk_stride,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
        )

    key = f"blanc-{args.type}-measure-{args.measure}"
    if args.doc is not None:
        result = model.eval_once(args.doc, args.summary)
        result_json = {key: result}
    elif args.single_json is not None:
        with open(args.single_json) as reader:
            data = json.load(reader)

        result = model.eval_once(data[args.doc_key], data[args.summary_key])
        result_json = {key: result}
    elif args.pairs_json is not None:
        with open(args.pairs_json) as reader:
            data = json.load(reader)
        docs = [pair[args.doc_key] for pair in data]
        summaries = [pair[args.summary_key] for pair in data]

        result = model.eval_pairs(docs, summaries)
        result_json = [{key: score} for score in result]
    elif args.doc_summaries_json is not None:
        with open(args.doc_summaries_json) as reader:
            data = json.load(reader)
        docs = [doc_summary[args.doc_key] for doc_summary in data]
        doc_summaries = [doc_summary[args.summaries_key] for doc_summary in data]

        result = model.eval_summaries_for_docs(docs, doc_summaries)
        result_json = [{key: scores} for scores in result]
    else:
        raise ValueError('Please provide an input document and summary')

    if args.output_json is None:
        print(result)
    else:
        with open(args.output_json, 'w') as writer:
            json.dump(result_json, writer)


if __name__ == '__main__':
    main()

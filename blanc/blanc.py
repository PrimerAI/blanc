import copy
import logging
import random

logging.getLogger('transformers').setLevel(level=logging.WARNING)

from nltk.tokenize import sent_tokenize
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm
from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AlbertForMaskedLM, AlbertTokenizer

from blanc.utils import (
    BertInput,
    Defaults,
    batch_data,
    mask_tokens_evenly,
    mask_tokens_randomly,
    get_input_tensors,
    determine_correctness,
    measure_relative,
    measure_improve,
    clean_text,
    truncate_list_of_lists,
    truncate_sentence_and_summary,
    set_seed,
    NOT_MASKED,
    TOKEN_TYPE_A,
    LABEL_IGNORE,
    BERT_MAX_TOKENS,
    TOKEN_REPLACE_RANGE,
)


class Blanc:
    """An abstract superclass containing shared functionality between BlancHelp and BlancTune.
    measure ('relative' or 'improve') is a choice of how the success of inference is measured.
    Add '-counts' to return also counts: 'relative-counts' or 'improve-counts'.
    """

    def __init__(
        self,
        model_name=Defaults.model_name,
        measure=Defaults.measure,
        gap=Defaults.gap,
        gap_mask=Defaults.gap_mask,
        gap_tune=Defaults.gap_tune,
        gap_mask_tune=Defaults.gap_mask_tune,
        min_token_length_normal=Defaults.min_token_length_normal,
        min_token_length_lead=Defaults.min_token_length_lead,
        min_token_length_followup=Defaults.min_token_length_followup,
        min_token_length_normal_tune=Defaults.min_token_length_normal_tune,
        min_token_length_lead_tune=Defaults.min_token_length_lead_tune,
        min_token_length_followup_tune=Defaults.min_token_length_followup_tune,
        device=Defaults.device,
        inference_batch_size=Defaults.inference_batch_size,
        inference_mask_evenly=Defaults.inference_mask_evenly,
        len_sent_allow_cut=Defaults.len_sent_allow_cut,
        p_mask=Defaults.p_mask,
        show_progress_bar=Defaults.show_progress_bar,
    ):
        """This class should not be instantiated directly: instead use BlancHelp or BlancTune"""
        self.model_name = model_name
        self.measure = measure
        self.gap = gap
        self.gap_mask = gap_mask
        self.gap_tune = gap_tune
        self.gap_mask_tune = gap_mask_tune
        self.min_token_length_normal = min_token_length_normal
        self.min_token_length_lead = min_token_length_lead
        self.min_token_length_followup = min_token_length_followup
        self.min_token_length_normal_tune = min_token_length_normal_tune
        self.min_token_length_lead_tune = min_token_length_lead_tune
        self.min_token_length_followup_tune = min_token_length_followup_tune
        self.device = device
        self.inference_batch_size = inference_batch_size
        self.inference_mask_evenly = inference_mask_evenly
        self.len_sent_allow_cut = len_sent_allow_cut
        self.p_mask = p_mask
        self.show_progress_bar = show_progress_bar

        # The same is intentionally not given:
        self.gap_tune = self.gap if self.gap_tune < 0 else self.gap_tune
        self.gap_mask_tune = self.gap_mask if self.gap_mask_tune < 0 else self.gap_mask_tune

        if self.model_name.lower().find('albert') >= 0:
            self.model_tokenizer = AlbertTokenizer.from_pretrained(model_name)
        else:
            self.model_tokenizer = BertTokenizer.from_pretrained(model_name)

    def eval_once(self, doc, summary):
        """Calculate the BLANC score for a single doc with a single summary.

        Args:
            doc (str): The input document
            summary (str): The input summary for the input document

        Returns:
            score (float): The BLANC score for the input
        """
        (doc_score,) = self.eval_summaries_for_docs([doc], [[summary]])
        (score,) = doc_score
        return score

    def eval_pairs(self, docs, summaries):
        """Calculate the BLANC score for multiple docs, each with a single summary

        Args:
            docs (List[str]): A list of input documents
            summaries (List[str]): The input summary for each input document

        Returns:
            scores (List[float]): The BLANC scores for the inputs
        """
        doc_summaries = [[summary] for summary in summaries]
        full_scores = self.eval_summaries_for_docs(docs, doc_summaries)
        scores = [score for score, in full_scores]
        return scores

    def eval_summaries_for_docs(self, docs, doc_summaries):
        """Calculate the BLANC score for multiple docs, each with multiple summaries

        Args:
            docs (List[str]): A list of input documents
            doc_summaries (List[List[str]]): A list of summaries for every input document

        Returns:
            scores (List[List[float]]): A list of blanc scores corresponding to each summary for
                each document
        """
        raise NotImplementedError()

    def get_inputs_for_sentence(self, sent_tokens, summary_tokens):
        """Used by subclasses to specify inference inputs corresponding to a sentence

        Args:
            sent_tokens (List[str]): list of tokens corresponding to sentence
            summary_tokens (List[str]): list of tokens corresponding to a summary
            sep (List[str]): List of tokens corresponding to a separator between summary and sentence

        Returns:
            inputs (List[BertInput]): a list of masked token inputs to BERT
            answers (List[Dict[int, str]]): a list of "answer" dicts, where each answer dict maps
                token indices corresponding to masked tokens back to their original token.
        """
        raise NotImplementedError()

    def mask_and_infer(self, model, docs, doc_summaries, sep=None):
        """Run the given model on masked versions of the provided doc_summaries and collect model
        output

        Args:
            model (BertForMaskedLM): a BERT for masked language modeling torch model
            docs (List[str]): A list of input documents
            doc_summaries (List[List[str]]): A list of summaries for every input document
            sep (str): Separator between the inference help (summary) and a sentence from the doc

        Returns:
            all_outputs (List[List[List[Dict[int, str]]]]): for each doc, for each summary for the
                doc, for each input sequence for the summary, we have a dict mapping indices to
                model predictions
            all_answers (List[List[List[Dict[int, str]]]]): for each doc, for each summary for the
                doc, for each input sequence for the summary, we have a dict mapping indices to
                original tokens
        """

        # Prepare inputs
        all_inputs, all_answers = [], []
        for doc, summaries in zip(docs, doc_summaries):
            doc_inputs, doc_answers = [], []
            for summary in summaries:
                summary_inputs, summary_answers = self.get_inference_inputs(doc, summary, sep)
                doc_inputs.append(summary_inputs)
                doc_answers.append(summary_answers)
            all_inputs.append(doc_inputs)
            all_answers.append(doc_answers)

        # Run inference in batches
        inputs_per_summary_per_doc = [
            [len(inputs) for inputs in summary_input] for summary_input in all_inputs
        ]
        collapsed_inputs = sum(sum(all_inputs, []), [])
        batched_inputs = batch_data(collapsed_inputs, self.inference_batch_size)

        iterator = tqdm.tqdm(batched_inputs, disable=not self.show_progress_bar)
        batched_outputs = [self.run_inference_batch(model, batch) for batch in iterator]
        collapsed_outputs = sum(batched_outputs, [])

        # Regroup outputs
        i = 0
        all_outputs = []
        for inputs_per_summary in inputs_per_summary_per_doc:
            doc_outputs = []
            for num_inputs in inputs_per_summary:
                doc_outputs.append(collapsed_outputs[i : i + num_inputs])
                i += num_inputs
            all_outputs.append(doc_outputs)

        return all_outputs, all_answers

    def get_inference_inputs(self, doc, summary=None, sep=None):
        """Get the inference inputs for a document, which possibly includes a summary

        Args:
            doc (str): an input document
            summary (str): an optional input summary
            sep (str): Separator between the inference help (summary) and a sentence from the doc

        Returns:
            summary_inputs (List[BertInput]): a list of BertInputs for inference
            summary_answers (List[Dict[int, str]]): each dict maps token indices back to their
                original token
        """
        doc = clean_text(doc)
        doc_sents = sent_tokenize(doc)
        doc_sent_tokens = [self.model_tokenizer.tokenize(sent) for sent in doc_sents]

        summary_sent_tokens = None
        if summary:
            summary = clean_text(summary)
            summary_sents = sent_tokenize(summary)
            summary_sent_tokens = [self.model_tokenizer.tokenize(sent) for sent in summary_sents]
        if not summary_sent_tokens:
            summary_sent_tokens = [[]]

        len_sep = 0
        if sep:
            len_sep = len(sep)

        summary_inputs, summary_answers = [], []
        half_num_sents = len(doc_sent_tokens)
        truncate_bottom = True
        for i_sent, sent_tokens in enumerate(doc_sent_tokens):
            if i_sent > half_num_sents:
                truncate_bottom = False
            sent_tokens, summary_tokens = truncate_sentence_and_summary(
                sent=sent_tokens,
                summary=summary_sent_tokens,
                len_sep=len_sep,
                len_sent_allow_cut=self.len_sent_allow_cut,
                truncate_bottom=truncate_bottom,
            )
            # now it is assured that everything fits into the allowed input size:
            assert len(sent_tokens) + len(summary_tokens) + len_sep + 2 <= BERT_MAX_TOKENS
            inputs, answers = self.get_inputs_for_sentence(sent_tokens, summary_tokens)
            summary_inputs += inputs
            summary_answers += answers
        return summary_inputs, summary_answers

    def assemble_inference_input(self, answers, sent_tokens, help_tokens=None, help_sep=None):
        """Given input tokens, assemble them into the tensors used by the model for inference

        Args:
            answers (Dict[int, str]): a mapping of input token indices to their original value
            sent_tokens (List[str]): tokens corresponding to an input sentence
            help_tokens (List[str]): tokens corresponding to an input summary or filler
            help_sep (List[str]): tokens to put between the summary/filler and the sentence

        Returns:
            model_input (BertInput): an input to the BERT model
            shifted_answers (Dict[int, str]): the input answers but with shifted indices that take
                into account the summary/filler and starting CLS token

        Raises:
            ValueError: if the sentence itself is longer than the BERT_MAX_TOKENS limit, we raise
                this error as opposed to truncating the sentence
        """
        if not help_tokens:
            help_tokens = []
        if not help_sep:
            help_sep = []

        all_tokens = (
            [self.model_tokenizer.cls_token]
            + help_tokens
            + help_sep
            + sent_tokens
            + [self.model_tokenizer.sep_token]
        )

        input_ids = self.model_tokenizer.convert_tokens_to_ids(all_tokens)
        token_type_ids = [TOKEN_TYPE_A] * len(all_tokens)
        attention_mask = [NOT_MASKED] * len(all_tokens)

        offset = 1 + len(help_tokens) + len(help_sep)
        shifted_answers = {}
        for idx, token in answers.items():
            shifted_answers[idx + offset] = token

        model_input = BertInput(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=None,
            masked_idxs=list(shifted_answers.keys()),
        )

        return model_input, shifted_answers

    def run_inference_batch(self, model, batch):
        """Run an inference batch through the provided model

        Args:
            model (BertForMaskedLM): a BERT for masked language modeling torch model
            batch (List[BertInput]): the input batch to run through the model

        Returns:
            all_predictions (List[Dict[int, str]]): predicted tokens for every masked token in
                the inputs
        """
        input_ids, attention_mask, token_type_ids, _ = get_input_tensors(
            batch, device=self.device, tokenizer=self.model_tokenizer,
        )

        with torch.no_grad():
            (model_output_batch,) = model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            )

        all_predictions = []
        for model_input, model_output in zip(batch, model_output_batch):
            predictions = {}
            for idx in model_input.masked_idxs:
                predicted_id = model_output[idx].argmax()
                (predicted_token,) = self.model_tokenizer.convert_ids_to_tokens([predicted_id])
                predictions[idx] = predicted_token
            all_predictions.append(predictions)

        return all_predictions

    def mask_input_tokens(self, tokens, is_finetune):
        """Given a list of tokens, produce maskings for them

        Args:
            tokens (List[str]): a sequence of wordpiece tokens
            is_finetune (bool): whether or not these tokens are going to be used for finetuning

        Returns:
            masked_inputs (List[List[str]]): a list of token sequences, where each token sequence
                contains some masked tokens.
            all_answers (List[Dict[int, str]]): a list of "answer" dicts, where each answer dict maps
                token indices corresponding to masked tokens back to their original token.
        """
        if is_finetune:
            even_masking = self.finetune_mask_evenly
        else:
            even_masking = self.inference_mask_evenly

        min_token_lengths = (
            self.min_token_length_normal,
            self.min_token_length_lead,
            self.min_token_length_followup,
        )
        if is_finetune:
            min_token_lengths = (
                self.min_token_length_normal_tune,
                self.min_token_length_lead_tune,
                self.min_token_length_followup_tune,
            )

        if even_masking:
            gap_use = self.gap
            gap_mask_use = self.gap_mask
            if is_finetune:
                gap_use = self.gap_tune
                gap_mask_use = self.gap_mask_tune
            return mask_tokens_evenly(
                tokens=tokens,
                gap=gap_use,
                min_token_lengths=min_token_lengths,
                mask_token=self.model_tokenizer.mask_token,
                gap_mask=gap_mask_use,
            )
        else:
            return mask_tokens_randomly(
                tokens=tokens,
                min_token_lengths=min_token_lengths,
                mask_token=self.model_tokenizer.mask_token,
                p_mask=self.p_mask,
            )

    def judge_output(self, base_output, assisted_output, base_answers, assisted_answers):
        """Given a model's predicted tokens with and without assistance, as well as the correct
        token predictions, produce the BLANC score

        Args:
            base_outputs (List[Dict[int, str]]): outputs without using "help" or "tune." Each list
                represents a different input masking, and each dict maps indices to model
                predictions.
            assisted_outputs (List[Dict[int, str]]): outputs using "help" or "tune." Each list
                represents a different input masking, and each dict maps indices to model
                predictions.
            base_answers (List[Dict[int, str]]): answers without using "help" or "tune." Each list
                represents a different input masking, and each dict maps indices to original
                tokens.
            assisted_answers (List[Dict[int, str]]): answers using "help" or "tune." Each
                list represents a different input masking, and each dict maps indices to original
                tokens.

        Returns:
            score (float): the BLANC score, if the measure is 'relative' or 'improve'.
            score, S (tuple of float and list): the BLANC score and counts,
                if the measure is 'relative-counts' or 'improve-counts'.
        """
        base_correctness = determine_correctness(base_output, base_answers)
        assisted_correctness = determine_correctness(assisted_output, assisted_answers)

        S = [[0, 0], [0, 0]]
        for base_correct, assisted_correct in zip(base_correctness, assisted_correctness):
            S[int(base_correct)][int(assisted_correct)] += 1

        measure_split = self.measure.split("-")
        if measure_split[0] == 'relative':
            result = measure_relative(S)
            if self.measure == 'relative-counts':
                result = result, S
        elif measure_split[0] == 'improve':
            result = measure_improve(S)
            if self.measure == 'improve-counts':
                result = result, S
        else:
            raise NotImplementedError()

        return result

    def init_model(self, device):
        """Initialize the language model and send it to the given device
        Note: Transformers v.4 and higher made default return_dict=True.
        Args:
            device (str): torch device (usually "cpu" or "cuda")

        Returns:
            model: a model for masked language modeling torch model
        """
        model = None
        if self.model_name.lower().find('albert') >= 0:
            try:
                model = AlbertForMaskedLM.from_pretrained(self.model_name, return_dict=False).to(device)
            except:
                model = AlbertForMaskedLM.from_pretrained(self.model_name).to(device)
        else:
            try:
                model = BertForMaskedLM.from_pretrained(self.model_name, return_dict=False).to(device)
            except:
                model = BertForMaskedLM.from_pretrained(self.model_name).to(device)
        model.eval()
        return model


class BlancHelp(Blanc):
    """BLANC-help, as defined in the BLANC paper."""

    def __init__(
        self,
        model_name=Defaults.model_name,
        measure=Defaults.measure,
        gap=Defaults.gap,
        gap_mask=Defaults.gap_mask,
        gap_tune=Defaults.gap_tune,
        gap_mask_tune=Defaults.gap_mask_tune,
        min_token_length_normal=Defaults.min_token_length_normal,
        min_token_length_lead=Defaults.min_token_length_lead,
        min_token_length_followup=Defaults.min_token_length_followup,
        min_token_length_normal_tune=Defaults.min_token_length_normal_tune,
        min_token_length_lead_tune=Defaults.min_token_length_lead_tune,
        min_token_length_followup_tune=Defaults.min_token_length_followup_tune,
        device=Defaults.device,
        inference_batch_size=Defaults.inference_batch_size,
        inference_mask_evenly=Defaults.inference_mask_evenly,
        len_sent_allow_cut=Defaults.len_sent_allow_cut,
        filler_token=Defaults.filler_token,
        help_sep=Defaults.help_sep,
        p_mask=Defaults.p_mask,
        show_progress_bar=Defaults.show_progress_bar,
    ):
        """See CLI documentation (blanc --help) for information about each arg"""
        super().__init__(
            model_name=model_name,
            measure=measure,
            gap=gap,
            gap_mask=gap_mask,
            gap_tune=gap_tune,
            gap_mask_tune=gap_mask_tune,
            min_token_length_normal=min_token_length_normal,
            min_token_length_lead=min_token_length_lead,
            min_token_length_followup=min_token_length_followup,
            min_token_length_normal_tune=min_token_length_normal_tune,
            min_token_length_lead_tune=min_token_length_lead_tune,
            min_token_length_followup_tune=min_token_length_followup_tune,
            device=device,
            inference_batch_size=inference_batch_size,
            inference_mask_evenly=inference_mask_evenly,
            len_sent_allow_cut=len_sent_allow_cut,
            p_mask=p_mask,
            show_progress_bar=show_progress_bar,
        )

        self.filler_token = filler_token
        self.help_sep = self.model_tokenizer.tokenize(help_sep)
        self.model = self.init_model(self.device)

    def eval_summaries_for_docs(self, docs, doc_summaries):
        """Calculate the BLANC score for multiple docs, each with multiple summaries.
        See documentation in superclass.
        """
        all_outputs, all_answers = self.mask_and_infer(
            self.model, docs, doc_summaries, sep=self.help_sep
        )

        all_scores = []
        for doc_outputs, doc_answers in zip(all_outputs, all_answers):
            doc_scores = []
            for summary_output, summary_answers in zip(doc_outputs, doc_answers):
                help_output = [out for i, out in enumerate(summary_output) if i % 2 == 0]
                filler_output = [out for i, out in enumerate(summary_output) if i % 2 == 1]
                help_answers = [answer for i, answer in enumerate(summary_answers) if i % 2 == 0]
                filler_answers = [answer for i, answer in enumerate(summary_answers) if i % 2 == 1]

                score = self.judge_output(filler_output, help_output, filler_answers, help_answers)
                doc_scores.append(score)
            all_scores.append(doc_scores)

        return all_scores

    def get_inputs_for_sentence(self, sent_tokens, summary_tokens):
        """Get inference inputs corresponding to a given sentence. For BLANC-help, we get several
        maskings for each sentence, and for each of these maskings we have an input with the
        summary prepended, and an input with a filler prepended. See documentation in superclass.
        """
        sent_maskings, init_answers = self.mask_input_tokens(sent_tokens, is_finetune=False)

        filler_tokens = [self.filler_token] * len(summary_tokens)
        inputs, final_answers = [], []
        for sent_masking, init_answer in zip(sent_maskings, init_answers):
            help_input, help_answers = self.assemble_inference_input(
                answers=init_answer,
                sent_tokens=sent_masking,
                help_tokens=summary_tokens,
                help_sep=self.help_sep,
            )

            filler_input, filler_answers = self.assemble_inference_input(
                answers=init_answer,
                sent_tokens=sent_masking,
                help_tokens=filler_tokens,
                help_sep=self.help_sep,
            )

            inputs += [help_input, filler_input]
            final_answers += [help_answers, filler_answers]

        return inputs, final_answers


class BlancTune(Blanc):
    """BLANC-tune, as defined in the BLANC paper."""

    def __init__(
        self,
        model_name=Defaults.model_name,
        measure=Defaults.measure,
        gap=Defaults.gap,
        gap_mask=Defaults.gap_mask,
        gap_tune=Defaults.gap_tune,
        gap_mask_tune=Defaults.gap_mask_tune,
        min_token_length_normal=Defaults.min_token_length_normal,
        min_token_length_lead=Defaults.min_token_length_lead,
        min_token_length_followup=Defaults.min_token_length_followup,
        device=Defaults.device,
        min_token_length_normal_tune=Defaults.min_token_length_normal_tune,
        min_token_length_lead_tune=Defaults.min_token_length_lead_tune,
        min_token_length_followup_tune=Defaults.min_token_length_followup_tune,
        inference_batch_size=Defaults.inference_batch_size,
        inference_mask_evenly=Defaults.inference_mask_evenly,
        finetune_batch_size=Defaults.finetune_batch_size,
        finetune_epochs=Defaults.finetune_epochs,
        finetune_mask_evenly=Defaults.finetune_mask_evenly,
        len_sent_allow_cut=Defaults.len_sent_allow_cut,
        finetune_chunk_size=Defaults.finetune_chunk_size,
        finetune_chunk_stride=Defaults.finetune_chunk_stride,
        finetune_top_fully=Defaults.finetune_top_fully,
        id_layer_freeze_below=Defaults.id_layer_freeze_below,
        id_layer_freeze_above=Defaults.id_layer_freeze_above,
        show_progress_bar=Defaults.show_progress_bar,
        p_mask=Defaults.p_mask,
        p_token_replace=Defaults.p_token_replace,
        p_token_original=Defaults.p_token_original,
        learning_rate=Defaults.learning_rate,
        warmup_steps=Defaults.warmup_steps,
        random_seed=Defaults.random_seed,
    ):
        """See CLI documentation (blanc --help) for information about each arg"""
        super().__init__(
            model_name=model_name,
            measure=measure,
            gap=gap,
            gap_mask=gap_mask,
            gap_tune=gap_tune,
            gap_mask_tune=gap_mask_tune,
            min_token_length_normal=min_token_length_normal,
            min_token_length_lead=min_token_length_lead,
            min_token_length_followup=min_token_length_followup,
            min_token_length_normal_tune=min_token_length_normal_tune,
            min_token_length_lead_tune=min_token_length_lead_tune,
            min_token_length_followup_tune=min_token_length_followup_tune,
            device=device,
            inference_batch_size=inference_batch_size,
            inference_mask_evenly=inference_mask_evenly,
            len_sent_allow_cut=len_sent_allow_cut,
        )

        self.finetune_batch_size = finetune_batch_size
        self.finetune_epochs = finetune_epochs
        self.finetune_mask_evenly = finetune_mask_evenly
        self.finetune_chunk_size = finetune_chunk_size
        self.finetune_chunk_stride = finetune_chunk_stride
        self.finetune_top_fully = finetune_top_fully
        self.id_layer_freeze_below = id_layer_freeze_below
        self.id_layer_freeze_above = id_layer_freeze_above
        self.show_progress_bar = show_progress_bar
        self.p_mask = p_mask
        self.p_token_replace = p_token_replace
        self.p_token_original = p_token_original
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.random_seed = random_seed

        # The same is intentionally not given:
        self.gap_tune = self.gap if self.gap_tune < 0 else self.gap_tune
        self.gap_mask_tune = self.gap_mask if self.gap_mask_tune < 0 else self.gap_mask_tune
        self.min_token_length_normal_tune = self.min_token_length_normal if self.min_token_length_normal_tune < 0 else self.min_token_length_normal_tune
        self.min_token_length_lead_tune = self.min_token_length_lead if self.min_token_length_lead_tune < 0 else self.min_token_length_lead_tune
        self.min_token_length_followup_tune = self.min_token_length_followup if self.min_token_length_followup_tune < 0 else self.min_token_length_followup_tune

        self.base_model = self.init_model(self.device)

    def eval_summaries_for_docs(self, docs, doc_summaries):
        """Calculate the BLANC score for multiple docs, each with multiple summaries.
        See documentation in superclass.
        Note that a summary should not be used in any ways for base outputs and base answers.
        When a finetuned model is used, the summary can used, meaning that 'help' and 'tune' versions of BLANC are put together.
        """

        doc_summaries_use = [[None for s in summs] for summs in doc_summaries]
        base_outputs, base_answers = self.mask_and_infer(self.base_model, docs, doc_summaries_use)

        finetuned_outputs, finetuned_answers = [], []
        model_cpu = self.init_model(device='cpu')
        for doc, summaries in tqdm.tqdm(zip(docs, doc_summaries), total=len(docs), disable=not self.show_progress_bar):
            finetuned_doc_outputs, finetuned_doc_answers = [], []
            for summary in summaries:
                model_copy = copy.deepcopy(model_cpu)
                finetuned_model = model_copy.to(self.device)
                self.finetune(finetuned_model, summary)

                (finetuned_summary_output,), (finetuned_summary_answer,) = self.mask_and_infer(
                    finetuned_model, [doc], [[None]]
                )
                finetuned_doc_outputs += finetuned_summary_output
                finetuned_doc_answers += finetuned_summary_answer

                del finetuned_model
                torch.cuda.empty_cache()

            finetuned_outputs.append(finetuned_doc_outputs)
            finetuned_answers.append(finetuned_doc_answers)

        all_scores = [
            [
                self.judge_output(
                    base_summary_output,
                    finetuned_summary_output,
                    base_summary_answers,
                    finetuned_summary_answers,
                )
                for (
                    base_summary_output,
                    base_summary_answers,
                    finetuned_summary_output,
                    finetuned_summary_answers,
                ) in zip(
                    base_doc_output, base_doc_answers, finetuned_doc_output, finetuned_doc_answers,
                )
            ]
            for (
                base_doc_output,
                base_doc_answers,
                finetuned_doc_output,
                finetuned_doc_answers,
            ) in zip(
                base_outputs, base_answers, finetuned_outputs, finetuned_answers,
            )
        ]

        return all_scores

    def get_inputs_for_sentence(self, sent_tokens, summary_tokens):
        """Get inference inputs corresponding to a given sentence. For BLANC-tune, we get several
        maskings for each sentence, and each masking is a single input. See documentation in
        superclass.
        """
        sent_maskings, init_answers = self.mask_input_tokens(sent_tokens, is_finetune=False)
        inputs, final_answers = [], []
        for sent_idx, (sent_masking, init_answer) in enumerate(zip(sent_maskings, init_answers)):
            input_, answers = self.assemble_inference_input(
                answers=init_answer, sent_tokens=sent_masking,
            )

            inputs.append(input_)
            final_answers.append(answers)

        return inputs, final_answers

    def finetune(self, model, summary):
        """Finetune the given model on a "dataset" produced from chunks of the given summary.

        Args:
            model (BertForMaskedLM): a BERT for masked language modeling torch model
            summary (str): the summary to finetune on
        """
        if self.random_seed > 0:
            set_seed(self.random_seed)
        model.train()
        n_params = len(list(model.parameters()))
        # Freeze a few lowest or a few highest layers:
        if self.id_layer_freeze_below > 0 or self.id_layer_freeze_above > 0:
            for i, param in enumerate(model.parameters()):
                if i < self.id_layer_freeze_below:
                    param.requires_grad = False
                elif self.id_layer_freeze_above < 0:
                    break
                elif n_params - i < self.id_layer_freeze_above:
                    param.requires_grad = False
        all_inputs = self.prepare_finetuning_data(summary)
        input_batches = batch_data(all_inputs, self.finetune_batch_size)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-2,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=len(input_batches) * self.finetune_epochs,
        )
        for epoch in range(self.finetune_epochs):
            for input_batch in input_batches:
                input_ids, attention_mask, token_type_ids, labels = get_input_tensors(
                    input_batch, device=self.device, tokenizer=self.model_tokenizer,
                )
                model.zero_grad()
                optimizer.zero_grad()
                try:  # masked_lm_labels were deprecated, replace by labels in transformers v4.x
                    loss, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )
                except:
                    loss, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        masked_lm_labels=labels,
                    )
                loss.backward()
                optimizer.step()
                scheduler.step()
        model.eval()

    def prepare_finetuning_data(self, summary):
        """Create a finetuning dataset using chunks of the given summary
        The finetune_top_fully=True compensate finetuning of top tokens, which
            otherwise get less tuning than tokens at further strides.

        Args:
            summary (str): the input summary to finetune on

        Returns:
            model_inputs (List[BertInput]): a list of inputs to use as the finetuning dataset
        """
        summary_tokens = self.model_tokenizer.tokenize(summary)
        model_inputs = []
        for start_token in range(0, len(summary_tokens), self.finetune_chunk_stride):
            end_token = start_token + self.finetune_chunk_size
            chunk_tokens = summary_tokens[start_token:end_token]
            model_inputs += self.assemble_finetuning_input(chunk_tokens)
            if self.finetune_top_fully and start_token > 0 and start_token < self.finetune_chunk_size:
                chunk_tokens = summary_tokens[:start_token]
                model_inputs += self.assemble_finetuning_input(chunk_tokens)
        return model_inputs

    def assemble_finetuning_input(self, chunk_tokens):
        """Given input tokens, assemble them into the tensors used by the model for finetuning

        Args:
            chunk_tokens (List[str]): a token sequence

        Returns:
            model_inputs (List[BertInput]): BertInputs corresponding to different maskings of
                chunk_tokens
        """
        all_input_tokens, all_answers = self.mask_input_tokens(chunk_tokens, is_finetune=True)

        all_input_tokens = [
            [self.model_tokenizer.cls_token] + tokens + [self.model_tokenizer.sep_token]
            for tokens in all_input_tokens
        ]
        all_input_ids = [
            self.model_tokenizer.convert_tokens_to_ids(tokens) for tokens in all_input_tokens
        ]
        all_labels = [[LABEL_IGNORE] * len(tokens) for tokens in all_input_tokens]

        model_inputs = []
        for input_ids, answers, labels in zip(all_input_ids, all_answers, all_labels):
            for original_idx, token in answers.items():
                idx = original_idx + 1  # accounting for starting CLS token
                (original_token_id,) = self.model_tokenizer.convert_tokens_to_ids([token])
                labels[idx] = original_token_id

                random_number = random.random()
                if random_number < self.p_token_replace:
                    # replace with a random token
                    input_ids[idx] = random.randint(*TOKEN_REPLACE_RANGE)
                elif random_number < self.p_token_original + self.p_token_replace:
                    # use original token
                    input_ids[idx] = original_token_id

            attention_mask = [NOT_MASKED] * len(input_ids)
            token_type_ids = [TOKEN_TYPE_A] * len(input_ids)
            model_input = BertInput(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                masked_idxs=None,
            )
            model_inputs.append(model_input)

        return model_inputs

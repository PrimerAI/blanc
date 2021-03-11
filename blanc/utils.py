from collections import namedtuple
import random
import unicodedata
import copy

import torch
from torch.nn.utils.rnn import pad_sequence

# prefix used by the wordpiece tokenizer to indicate that the token continues the previous word
WORDPIECE_PREFIX = '##'
# a range of reasonable token ids to use for replacement during model training
TOKEN_REPLACE_RANGE = (1000, 29999)
# attention mask value that tells the model to not ignore the token
NOT_MASKED = 1
# token type for a single input sequence
TOKEN_TYPE_A = 0
# padding value used for attention_mask
MASK_PAD = 0
# padding value used for token_type_ids
TOKEN_TYPE_PAD = 1
# label to use for tokens to ignore when computing masked language modeling training loss
LABEL_IGNORE = -100
# maximum number of input tokens BERT supports
BERT_MAX_TOKENS = 512

# used to represent inputs to the BERT model
BertInput = namedtuple(
    typename='BertInput',
    field_names=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'masked_idxs'],
)

# all the configuration options
Config = namedtuple(
    'Config',
    [
        'doc_key',
        'summary_key',
        'summaries_key',
        'model_name',
        'measure',
        'gap',
        'gap_mask',
        'gap_tune',
        'gap_mask_tune',
        'min_token_length_normal',
        'min_token_length_lead',
        'min_token_length_followup',
        'min_token_length_normal_tune',
        'min_token_length_lead_tune',
        'min_token_length_followup_tune',
        'device',
        'random_seed',
        'inference_batch_size',
        'inference_mask_evenly',
        'len_sent_allow_cut',
        'filler_token',
        'help_sep',
        'finetune_batch_size',
        'finetune_epochs',
        'finetune_mask_evenly',
        'finetune_chunk_size',
        'finetune_chunk_stride',
        'finetune_top_fully',
        'id_layer_freeze_below',
        'id_layer_freeze_above',
        'show_progress_bar',
        'p_mask',
        'p_token_replace',
        'p_token_original',
        'learning_rate',
        'warmup_steps',
    ],
)

# the default configuration options that don't require a GPU
# We found gap=2 to work the best. To reproduce the original paper results use gap=6 
Defaults = Config(
    doc_key='doc',
    summary_key='summary',
    summaries_key='summaries',
    model_name='bert-base-uncased',
    measure='relative',
    gap=2,
    gap_mask=1,
    gap_tune=-1,
    gap_mask_tune=-1,
    min_token_length_normal=4,
    min_token_length_lead=2,
    min_token_length_followup=100,
    min_token_length_normal_tune=-1,
    min_token_length_lead_tune=-1,
    min_token_length_followup_tune=-1,
    device='cpu',
    random_seed=0,
    inference_batch_size=1,
    inference_mask_evenly=True,
    len_sent_allow_cut=100,
    filler_token='.',
    help_sep='',
    finetune_batch_size=1,
    finetune_epochs=10,
    finetune_chunk_size=64,
    finetune_chunk_stride=32,
    finetune_top_fully=True,
    id_layer_freeze_below=-1,
    id_layer_freeze_above=-1,
    show_progress_bar=True,
    p_mask=0.15,
    p_token_replace=0.1,
    p_token_original=0.1,
    learning_rate=5e-5,
    finetune_mask_evenly=True,
    warmup_steps=0,
)


def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)


def batch_data(data, batch_size):
    """Given a list, batch that list into chunks of size batch_size

    Args:
        data (List): list to be batched
        batch_size (int): size of each batch

    Returns:
        batches (List[List]): a list of lists, each inner list of size batch_size except possibly
            the last one.
    """
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    return batches


def is_token_large_enough(token, next_token, min_token_lengths):
    """Determine if a token is large enough according to min_token_lengths

    Args:
        token (str): a wordpiece token
        next_token (str): the next wordpiece token in the sequence
        min_token_lengths (Tuple[int, int, int]): minimum token lengths for normal tokens, lead
            tokens, and followup tokens

    Returns:
        large_enough (bool): whether or not the token is large enough
    """
    min_normal, min_lead, min_followup = min_token_lengths
    token_size = len(token)

    if token.startswith(WORDPIECE_PREFIX):
        token_size -= len(WORDPIECE_PREFIX)
        return token_size >= min_followup
    elif next_token.startswith(WORDPIECE_PREFIX):
        return token_size >= min_lead
    else:
        return token_size >= min_normal


def mask_tokens_evenly(tokens, gap, min_token_lengths, mask_token, gap_mask=1):
    """Produce several maskings for the given tokens where each masking is created by masking every
    "gap" tokens, as long as the token is large enough according to min_token_lengths.

    Args:
        tokens (List[str]): a sequence of wordpiece tokens
        gap (int): the spacing in-between masked tokens
        min_token_lengths (Tuple[int, int, int]): minimum token lengths for normal tokens, lead
            tokens, and followup tokens
        mask_token (str): wordpiece token to use for masking

    Returns:
        masked_inputs (List[List[str]]): a list of token sequences, where each token sequence
            contains masked tokens separated by "gap" tokens.
        all_answers (List[Dict[int, str]]): a list of "answer" dicts, where each answer dict maps
            token indices corresponding to masked tokens back to their original token.
    """
    gap = min(gap, len(tokens))
    masked_inputs = []
    all_answers = []
    for modulus in range(gap):
        masked_input = []
        answers = {}
        for idx, token in enumerate(tokens):
            next_token = '' if idx + 1 == len(tokens) else tokens[idx + 1]
            large_enough = is_token_large_enough(token, next_token, min_token_lengths)

            idx_off = idx % gap
            if gap == 1:
                can_mask = True
            elif modulus + gap_mask >= gap:
                can_mask = idx_off >= modulus or idx_off < (modulus + gap_mask)%gap
            else:
                can_mask = idx_off >= modulus and idx_off < modulus + gap_mask
            if can_mask and large_enough:
                masked_input.append(mask_token)
                answers[idx] = token
            else:
                masked_input.append(token)

        if len(answers) > 0:
            masked_inputs.append(masked_input)
            all_answers.append(answers)

    return masked_inputs, all_answers


def mask_tokens_randomly(tokens, min_token_lengths, mask_token, p_mask):
    """Produce several maskings for the given tokens by randomly choosing tokens to mask

    Args:
        tokens (List[str]): a sequence of wordpiece tokens
        min_token_lengths (Tuple[int, int, int]): minimum token lengths for normal tokens, lead
            tokens, and followup tokens
        mask_token (str): wordpiece token to use for masking

    Returns:
        masked_inputs (List[List[str]]): a list of token sequences, where each token sequence
            contains masked tokens chosen randomly.
        all_answers (List[Dict[int, str]]): a list of "answer" dicts, where each answer dict maps
            token indices corresponding to masked tokens back to their original token.
    """
    n_mask = max(int(len(tokens) * p_mask), 1)

    token_positions = []
    for idx, token in enumerate(tokens):
        next_token = '' if idx + 1 == len(tokens) else tokens[idx + 1]
        if is_token_large_enough(token, next_token, min_token_lengths):
            token_positions.append(idx)
    random.shuffle(token_positions)

    all_inputs, all_answers = [], []
    while len(token_positions) > 0:
        positions_to_mask = token_positions[:n_mask]
        token_positions = token_positions[n_mask:]

        inputs, answers = [], {}
        for idx, token in enumerate(tokens):
            if idx in positions_to_mask:
                inputs.append(mask_token)
                answers[idx] = token
            else:
                inputs.append(token)

        all_inputs.append(inputs)
        all_answers.append(answers)

    return all_inputs, all_answers


def stack_tensor(input_list, pad_value, device):
    """Given a batch of inputs, stack them into a single tensor on the given device, padding them
    at the back with pad_value to make sure they are all the same length.

    Args:
        input_list (List[List[int]]): a list of input sequences
        pad_value (int): the value to use for padding input sequences to make them the same length
        device (str): torch device (usually "cpu" or "cuda")

    Returns:
        stacked_tensor (torch.LongTensor): a tensor of dimensions (batch size) x (seq length)
    """
    tensor_list = [torch.LongTensor(inputs) for inputs in input_list]
    stacked_tensor = pad_sequence(
        sequences=tensor_list, batch_first=True, padding_value=pad_value
    ).to(device)

    return stacked_tensor


def get_input_tensors(input_batch, device, tokenizer):
    """Given a list of BertInputs, return the relevant tensors that are fed into BERT.

    Args:
        input_batch (List[BertInput]): a batch of model inputs
        device (str): torch device (usually "cpu" or "cuda")
        tokenizer (BertTokenizer): the wordpiece tokenizer used for BERT

    Returns:
        input_ids (torch.LongTensor): ids corresponding to input tokens
        attention_mask (torch.LongTensor): tells BERT about parts of the input to ignore
        token_type_ids (torch.LongTensor): used to differentiate input segments
        labels (torch.LongTensor): contains the original token ids for tokens that were masked
    """
    input_ids_list = [inputs.input_ids for inputs in input_batch]
    attention_mask_list = [inputs.attention_mask for inputs in input_batch]
    token_type_ids_list = [inputs.token_type_ids for inputs in input_batch]
    labels_list = [inputs.labels for inputs in input_batch]

    (id_pad,) = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])
    input_ids = stack_tensor(input_ids_list, pad_value=id_pad, device=device)
    attention_mask = stack_tensor(attention_mask_list, pad_value=MASK_PAD, device=device)
    token_type_ids = stack_tensor(token_type_ids_list, pad_value=TOKEN_TYPE_PAD, device=device)

    if labels_list[0] is not None:
        labels = stack_tensor(labels_list, pad_value=LABEL_IGNORE, device=device)
    else:
        labels = None

    return input_ids, attention_mask, token_type_ids, labels


def determine_correctness(outputs, answers):
    """Given dicts corresponding to predicted tokens and actual tokens at different indices, return
    a list of bools for whether or not those predictions were correct.

    Args:
        outputs (List[Dict[int, str]]): each list represents a different input masking, and each
            dict maps indices to model predictions
        answers (List[Dict[int, str]]): each list represents a different input masking, and each
            dict maps indices to original tokens

    Returns:
        correctness (List[bool]): a list of values that are True if the model made a correct
            prediction and False otherwise
    """
    correctness = []
    for output, answer in zip(outputs, answers):
        for idx, actual_token in answer.items():
            predicted_token = output[idx]
            correctness.append(predicted_token == actual_token)

    return correctness


def measure_relative(S):
    """Calculate the "measure-relative" score as defined in the paper

    Args:
        S (List[List[int]]): accuracy counts as defined in the paper

    Returns:
        score (float): measure-relative score
    """
    denom = S[0][0] + S[1][1] + S[0][1] + S[1][0]
    if denom == 0:
        return 0
    return (S[0][1] - S[1][0]) / denom


def measure_improve(S):
    """Calculate the "measure-improve" score as defined in the paper

    Args:
        S (List[List[int]]): accuracy counts as defined in the paper

    Returns:
        score (float): measure-improve score
    """
    denom = S[0][0] + S[1][1] + S[0][1]
    if denom == 0:
        return 0
    return S[0][1] / denom


def clean_text(text):
    """Return a cleaned version of the input text

    Args:
        text (str): dirty text

    Returns:
        text (str): cleaned text
    """
    text = unicodedata.normalize('NFKD', text)
    return text


def truncate_sentence_and_summary(
    sent, summary, len_sep=0, len_sent_allow_cut=0, truncate_bottom=True,
):
    """Cut summary+sentence to allowed input size. 2 more tokens: [CLS], [SEP]
    The summary must have at least one sublist (can be empty)
    The sentence is cut by tokens from the bottom.
    The summary is cut by sentences. Last sentence is cut by tokens.

    Args:
        sent (List[str]): Sentence as a list of tokens
        summary (List[List[str]]): Summary as list of sentences, each sentence is list of tokens
        len_sep (int): Number of tokens in a separator used between the summary and the sentence
        len_sent_allow_cut (int): Allowed size of truncated sentence before cutting summary
        truncate_bottom (bool): Indicator how to cut the summary

    Returns:
        sent (List[str]): Truncated (if necessary) sentence as a list of tokens
        summary_tokens (List[str]): Truncated (if necessary) summary as a list of tokens
    """
    summary_tokens = [t for sublist in summary for t in sublist]
    len_input_estimate = 2 + len(summary_tokens) + len_sep + len(sent)
    len_excess = len_input_estimate - BERT_MAX_TOKENS
    if len_excess > 0:
        len_cut_sent = min(len_excess, len(sent) - len_sent_allow_cut)
        len_sent_new = len(sent) - len_cut_sent
        sent = sent[:len_sent_new]
        assert len_excess <= len_cut_sent or summary[0]
        if len_excess > len_cut_sent:
            len_summary_max = BERT_MAX_TOKENS - 2 - len_sep - len(sent)
            summary_truncated = truncate_list_of_lists(
                sents_tokenized=summary, num_max=len_summary_max, truncate_bottom=truncate_bottom,
            )
            summary_tokens = [t for sublist in summary_truncated for t in sublist]
    assert len(sent) + len(summary_tokens) + len_sep + 2 <= BERT_MAX_TOKENS
    return sent, summary_tokens


def truncate_list_of_lists(sents_tokenized, num_max, truncate_bottom=True):
    """Return a truncated list, with summ of tokens not exceeding maximum.
    Truncate by lists. If single left list is still too long, truncate it by tokens.
    In our context each element of sents_tokenized is a sentence represented as a list of tokens.

    Args:
        sents_tokenized (List[List[str]]): List, each element is a list.
        num_max (int): maximal allowed number of tokens.
        truncate_bottom (bool): truncate starting from bottom lists.

    Returns:
        sents_tokenized (List[str]): truncated list
    """
    sents_truncated = []
    if truncate_bottom:
        len_truncated = 0
        # Cut by sentences:
        for sent in sents_tokenized:
            len_truncated_maybe = len_truncated + len(sent)
            if len_truncated_maybe > num_max:
                break
            len_truncated = len_truncated_maybe
            sents_truncated.append(sent)
            if len_truncated == num_max:
                break
    else:
        sents_truncated = copy.deepcopy(sents_tokenized)
        len_truncated = sum([len(s) for s in sents_tokenized])
        # Cut by sentences:
        for sent in sents_tokenized:
            if len_truncated <= num_max:
                break
            sents_truncated = sents_truncated[1:]
            len_truncated = len_truncated - len(sent)
    if not sents_truncated:
        sent_use = sents_tokenized[0] if truncate_bottom else sents_tokenized[-1]
        sents_truncated = [copy.deepcopy(sent_use)]
        len_truncated = len(sents_truncated[0])
        # Cut by tokens - always from the top:
        if len_truncated > num_max:
            len_remove = len_truncated - num_max
            sents_truncated[0] = sents_truncated[0][len_remove:]
    assert sum([len(s) for s in sents_truncated]) <= num_max
    return sents_truncated

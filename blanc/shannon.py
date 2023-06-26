import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    XLNetLMHeadModel,
    XLNetTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    ReformerModelWithLMHead,
    ReformerTokenizer,
    XLMWithLMHeadModel,
    XLMTokenizer,
)
import numpy as np
from nltk import sent_tokenize

def get_model(name, size, device='cuda'):
    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'gpt2':
        t = GPT2Tokenizer.from_pretrained('gpt2')
        if size == 'base':
            g = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            g = GPT2LMHeadModel.from_pretrained(f'gpt2-{size}')
        eos = g.config.eos_token_id # '<|endoftext|>'
        max_input = 1024
    elif name == 'gpt1':
        t = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        g = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        eos = 1 # .
        max_input = 512
    elif name == 'xlnet':
        t = XLNetTokenizer.from_pretrained(f'xlnet-{size}-cased')
        g = XLNetLMHeadModel.from_pretrained(f'xlnet-{size}-cased')
        eos = g.config.eos_token_id # </s>
        max_input = 1024
    elif name == 'transformerxl':
        t = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        g = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        eos = g.config.eos_token_id # <eos>
        max_input = 1024
    elif name == 'reformer':
        t = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
        g = ReformerModelWithLMHead.from_pretrained('google/reformer-crime-and-punishment')
        eos = g.config.eos_token_id # (empty string)
        max_input = 1024
    elif name == 'xlm':
        t = XLMTokenizer.from_pretrained('xlm-clm-ende-1024')
        g = XLMWithLMHeadModel.from_pretrained('xlm-clm-ende-1024')
        g.config.lang_id = g.config.lang2id['en']
        eos = 4 # </s>
        max_input = 1024

    g = g.to(device)
    g.config.return_dict = False
    g.eval()
    return g, t, eos, max_input

def prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    """Copied from gpt2 of huggingface transformers, but using it here separately, 
    because in some versions this funciton worked differently, which caused errors."""
    if past:  # only last token for inputs_ids if past is defined in kwargs
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None
    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }

class Shannon:
    def __init__(
        self,
        verbose=False,
        language_model='gpt2',
        model_size='base',
        num_upstream=0,
        return_token_lls=False,
        device='cuda'
    ):
        self.verbose = verbose
        self.language_model = language_model
        self.num_upstream = num_upstream
        self.return_token_lls = return_token_lls
        self.device = device
        if self.device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.g, self.t, self.eos, self.max_input = get_model(
            language_model, model_size, self.device)

    def measure(self, doc_tokens, prompt):
        eos = torch.LongTensor([self.eos]).to(self.device)
        if prompt is None or (prompt.dim() == 1 and len(prompt) == 0):
            prompt = torch.LongTensor([]).to(self.device)

        token_lls = []
        success = []
        past = None
        for i, token in enumerate(doc_tokens):
            upstream = doc_tokens[:i]
            if len(upstream) + len(prompt) + 1 > self.max_input:
                upstream = upstream[-(self.max_input - 1 - len(prompt)):]
                if past is not None:
                    past = [t[:, :, :, 1:, :] for t in past]

            prefix = torch.cat([eos, prompt, upstream]).unsqueeze(0)
            inputs = prepare_inputs_for_generation(prefix, past=past, use_cache=True, use_mems=True)
            with torch.no_grad():
                out = self.g(**inputs)

            if self.language_model in 'gpt2':
                logits, past = out
            elif self.language_model in ['gpt1', 'reformer']:
                logits, = out
                logits = logits[0, -1, :]
            elif self.language_model == 'xlnet':
                logits, = out
            elif self.language_model == 'transformerxl':
                logits, past = out
                logits = logits[0, -1, :]
            elif self.language_model == 'xlm':
                logits, = out
                logits = logits[0, -1, :]
            probs = F.softmax(logits, dim=-1).view(-1)
            prob = probs[token].item()

            log_prob = np.log(prob)
            token_lls.append(log_prob)
            success.append(int(token == probs.argmax()))

            true_token = self.t.decode([token])
            try:
                pred_token = self.t.decode([probs.argmax()])
            except:
                pred_token = None
            info = -log_prob / np.log(2)
            self.log(f'{true_token},{info}')

        return token_lls, success

    def go(self, doc, summ, measure_t=False, measure_summ=False):
        sents = sent_tokenize(doc)
        encode_args = {'return_tensors': 'pt'}
        if self.language_model == 'transformerxl':
            encode_args['add_space_before_punct_symbol'] = True
        if self.language_model in ['xlnet', 'xlm']:
            encode_args['add_special_tokens'] = False

        sents_tokens = [self.t.encode(sent, **encode_args).to(self.device).view(-1) for sent in sents]
        summ_tokens = self.t.encode(summ, **encode_args).to(self.device).view(-1)
        sents_tokens = [sent_tokens[:self.max_input - 1 - len(summ_tokens)] for sent_tokens in sents_tokens]
        doc_tokens = torch.cat(sents_tokens, dim=-1)

        if measure_t:
            ll, tries, success = 0, 0, []
            for sent_tokens in sents_tokens:
                sent_ll, sent_tries, sent_success = self.measure(sent_tokens, sent_tokens)
                ll += sent_ll
                tries += sent_tries
                success += sent_success
            return ll, tries, success

        elif measure_summ:
            summ_ll, summ_success = self.measure(summ_tokens, None)
            return summ_ll

        else:
            token_lls_base, token_lls_help, token_lls_full = [], [], []
            S = [[0, 0], [0, 0]]
            for sent_idx in range(len(sents_tokens)):
                sent_tokens = sents_tokens[sent_idx]
                upstream_tensors = sents_tokens[sent_idx-self.num_upstream:sent_idx]
                if len(upstream_tensors) > 0:
                    upstream_context = torch.cat(upstream_tensors)
                else:
                    upstream_context = torch.LongTensor([])
                    if self.device == 'cuda':
                        upstream_context = upstream_context.cuda()

                base_prompt = upstream_context
                help_prompt = torch.cat([summ_tokens, upstream_context])
                full_prompt = torch.cat([upstream_context, sent_tokens, upstream_context])

                base_sent_lls, base_sent_success = self.measure(sent_tokens, base_prompt)
                help_sent_lls, help_sent_success = self.measure(sent_tokens, help_prompt)
                full_sent_lls, full_sent_success = self.measure(sent_tokens, full_prompt)

                token_lls_base += base_sent_lls
                token_lls_help += help_sent_lls
                token_lls_full += full_sent_lls

                for b, h in zip(base_sent_success, help_sent_success):
                    S[b][h] += 1

            if self.return_token_lls:
                doc_tokens = self.t.convert_ids_to_tokens(doc_tokens)
                summ_tokens = self.t.convert_ids_to_tokens(summ_tokens)
                return token_lls_base, token_lls_help, token_lls_full, doc_tokens, summ_tokens
            else:
                ll_base = sum(token_lls_base)
                ll_help = sum(token_lls_help)
                ll_full = sum(token_lls_full)

                self.log(f'Shannon Score: {(ll_help - ll_base) / (ll_full - ll_base)}')
                self.log(f'Info Diff: {ll_help - ll_base}')
                self.log(f'BLANC: {(S[0][1] - S[1][0]) / (S[0][0] + S[0][1] + S[1][0] + S[1][1])}')

                return ll_base, ll_help, ll_full, S, len(doc_tokens), len(summ_tokens)

    def log(self, s=None):
        if self.verbose:
            print(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--measure_t', action='store_true')
    parser.add_argument('--measure_summ', action='store_true')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--eval', type=str, choices=['6', '47', '23'], default=None)
    parser.add_argument('--system', type=str, default=None)
    parser.add_argument('--lm', type=str, default='gpt2')
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--num_upstream', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    s = Shannon(args.verbose, args.lm, args.model_size, args.num_upstream)

    if args.simple:
        doc = 'Jack drove his minivan to the bazaar to purchase milk and honey for his large family'
        summ = 'Jack bought milk and honey from the bazaar'
        results = s.go(doc, summ, measure_t=args.measure_t, measure_summ=args.measure_summ)

        print(results)
    else:
        with open(args.input_file) as reader:
            if args.input_file.endswith('.jsonl'):
                data = [json.loads(line) for line in reader]
            else:
                data = json.load(reader)

        selection = data[args.start:]
        if args.eval is not None:
            selection = [record for record in data if record['eval'] in args.eval]
        if args.system is not None:
            selection = [record for record in selection if record['model_id'] == args.system]

        for record in tqdm(selection):
            if args.measure_t or args.measure_summ:
                ll = s.go(
                    record['doc'], record['summ'],
                    measure_summ=args.measure_summ, measure_t=args.measure_t
                )
                print(json.dumps({
                    'doc_id': record['id'],
                    'system': record['model_id'],
                    'll_summ': ll,
                }))

            else:
                ll_base, ll_help, ll_full, S, num_doc_tokens, num_summ_tokens = s.go(
                    record['doc'], record['summ']
                )
                print(json.dumps({
                    'doc_id': record['id'],
                    'system': record['model_id'],
                    'll_base': ll_base,
                    'll_help': ll_help,
                    'll_full': ll_full,
                    'num_doc_tokens': num_doc_tokens,
                    'num_summ_tokens': num_summ_tokens,
                    'S': S,
                }))

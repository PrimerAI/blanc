import unicodedata
import copy
import scipy
from numpy import dot
from numpy.linalg import norm

import logging
logging.getLogger('transformers').setLevel(level=logging.WARNING)

import nltk
from nltk.tokenize import word_tokenize

import torch
from transformers import BertForMaskedLM, BertTokenizer, BertModel


class Estime:
    """Estimator of factual inconsistencies between summaries (or other textual claims)
    and text. Usage: create `Estime`, and use `evaluate_claims()`.
    In creating Estime, specify the names of the desired measures in the list 'output'. 
    The function evaluate_claims() will return (for each claim) the list of results in 
    the same order. The list 'output' can also include:
        'alarms': the original ESTIME
        'alarms_adjusted': the original ESTIME, extrapolated to non-overlapping tokens
        'alarms_alltokens': ESTIME on all (not only overlapped) summary tokens
        'soft': the soft ESTIME
        'coherence': measure of summary coherence
    """
    def __init__(
        self,
        path_mdl='bert-large-uncased-whole-word-masking',
        path_mdl_raw='bert-base-uncased',
        i_layer_context=21,
        device='cpu',
        output=['alarms'],
        tags_check=None,
        tags_exclude=None,
        input_size_max=450,
        margin=50,
        distance_word_min=8):
        """  
        Args:
            path_mdl (str): model for embeddings of masked tokens
            path_mdl_raw (str): model for raw embeddings
            i_layer_context (int): index of layer to take contextual embeddings from
            device (str): 'cpu' or 'cuda'
            tags_check (List[str]): list of parts of speech to use, each is a tag
                by NLTK notations. If None, then all words will be used, 
                no matter which parts of speech they are.
            tags_exclude (List[str]): List or set of part-of-speach tags that should
                not be considered. Has priority over tags_check.
            input_size_max (int): length of input for the model as number of tokens
            margin (int): number of tokens on input edges not to take embeddings from
            distance_word_min (int): minimal distance between the masked tokens 
                from which embeddings are to ne taken in a single input
            output (List[str]): list of names of measures to get in claim evaluations. 
                The list must be nonempty and can include:
                'alarms', 'alarms_adjusted', 'alarms_alltokens', 'soft', 'coherence'.
                The function evaluate_claims() will return (for each claim) the list of 
                results in the same order.
        """
        self.i_layer_context = i_layer_context
        self.device = device
        assert output
        self.output = output
        self.ESTIME_ALARMS = 'alarms'  # original estime: by response of tokens to similar contexts
        self.ESTIME_ALLTOKENS = 'alarms_alltokens'  # estime on all (not only overlapped) summary tokens
        self.ESTIME_ADJUSTED = 'alarms_adjusted'  # original estime, extrapolated to non-overlapping tokens
        self.ESTIME_SOFT = 'soft'  # soft estime by response of similarity between embeddings
        self.ESTIME_COHERENCE = 'coherence'  # estimation of summary coherence
        self.get_estime_adjusted = self.ESTIME_ADJUSTED in self.output
        self.get_estime_soft = self.ESTIME_SOFT in self.output
        self.get_estime_coherence = self.ESTIME_COHERENCE in self.output
        self.tags_check = tags_check
        self.tags_exclude = tags_exclude
        self.input_size_max = input_size_max
        self.margin = margin
        self.distance_word_min = distance_word_min
        self.model_tokenizer = None
        self.model = None
        self.model_tokenizer = BertTokenizer.from_pretrained(path_mdl)
        self.model = BertForMaskedLM.from_pretrained(path_mdl, output_hidden_states = True).to(self.device)
        self.model.eval()
        if self.get_estime_soft:
            self.model_raw = BertModel.from_pretrained(path_mdl_raw)
            self.model_raw.eval()
            for param in self.model_raw.parameters():
                param.requires_grad = False
            self.embeddings_raw = self.model_raw.get_input_embeddings()
        # Convenient data (including tokenized summary and text):
        self.text_map_wordscheck = None
        self.summ_toks = None 
        self.text_toks = None
        self.embs_mask_text = None
        self.embs_raw_text = None


    def evaluate_claims(self, text, claims):
        """
        Given a text, and a list of claims (e.g. summaries, or other texts), 
        estimates how many likely factual inconsistencies each claim contains
        (the inconsistencies are with respect to the text).
        Returns for each claim whatever is specified in self.output
        Args:
            text (str): the main text
            claims (List[str]): texts ('claims') to be checked for consistency 
                with the main text. Each claim is preferably a shorter text, 
                like a claim/statement/summary.
        Returns: 
            claims_info: list of the same length as claims; each element is a
                consistency info for the corresponding claim. The info is a list
                accordingly to the names in self.output.
        """
        # Text:
        text = unicodedata.normalize('NFKD', text)
        text_words = word_tokenize(text)
        text_tagged = nltk.pos_tag(text_words)
        # Find all words of interest in the text, tokenize:
        self.text_map_wordscheck = self._get_map_words_intext(text_tagged)
        text_iwordscheck = sorted([item for sublist in self.text_map_wordscheck.values() for item in sublist])
        self.text_toks, text_map_iword_itoks = self._translate_words_to_tokens(text_words)
        # All embeddings of interest in the text:
        self.embs_mask_text = self._get_embeddings(
            tokens=self.text_toks, 
            ixs_words=text_iwordscheck, 
            map_words_to_tokens=text_map_iword_itoks)
        self.embs_raw_text = self._get_embeddings_raw(tokens=self.text_toks)
        # Get the consistency info for each claim:
        claims_info = []
        for claim in claims:
            claim = unicodedata.normalize('NFKD', claim)
            claim_info = self._evaluate_claim(claim)
            claims_info.append(claim_info)
        self.summ_toks = None 
        self.text_toks = None
        return claims_info


    def _evaluate_claim(self, claim, words_check=None):
        """
        Text is already processed, its embeddings can be used for the claim.
        Args:
            claim (str): claim, e.g. summary or short text - not the main text
            words_check (Set{str}): a set or map where the keys are the words 
                of interest in the main text
        Returns:
            estime_info (List[float]): a list with results corresponding to the 
                names of measures specified in self.output.
        """
        summ_words = word_tokenize(claim)
        summ_tagged = nltk.pos_tag(summ_words)
        summ_iwordscheck, summ_iwords_overlap = [],[]  # Find all words of interest in the summary
        for i, (w, t) in enumerate(summ_tagged):
            if not words_check or w in words_check:  # if required, checking only what exists in the text
                summ_iwordscheck.append(i)
            if not self.text_map_wordscheck or w in self.text_map_wordscheck:
                summ_iwords_overlap.append(i)
        self.summ_toks, summ_map_iword_itoks = self._translate_words_to_tokens(summ_words)
        embs_mask_summ = self._get_embeddings(
            tokens=self.summ_toks, ixs_words=summ_iwordscheck, map_words_to_tokens=summ_map_iword_itoks)
        embs_raw_summ = self._get_embeddings_raw(tokens=self.summ_toks)
        summ_itoksoverlap = set()
        for iword in summ_iwords_overlap:
            itok = summ_map_iword_itoks[iword][0]  # only first token of each word
            summ_itoksoverlap.add(itok)
        estime_info = self._evaluate(
            embs_mask_summ, self.embs_mask_text, 
            embs_raw_summ, self.embs_raw_text, summ_itokscheck=summ_itoksoverlap)
        return estime_info


    def _get_embeddings_raw(self, tokens):
        """Simply gets raw embeddings. Needed only for estime-soft."""
        if not self.get_estime_soft:
            return None
        toks_ids = self.model_tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.LongTensor([toks_ids])
        word_embeddings = self.embeddings_raw(input_tensor)[0].numpy()
        return word_embeddings


    def _get_embeddings(self, tokens, ixs_words, map_words_to_tokens):
        """
        Finds embeddings for all tokens of all words of interest. The embeddings 
        are obtained one group of words at a time; each group contains well 
        separated indexes, so that masked indexes do have enough context around.
        Args:
            tokens (List[str]): List of tokens, as strings. Represents the text.
            ixs_words (List[int]): List of indexes of words to check
            map_words_to_tokens Dict[int:(int,int)]: 
                Maps each index of word in the text to its range of tokens, the 
                range is: index of first token, index of one-past-last token.
        Returns:
            map_itok_embeddings (Dict{int: ndarray[float]}): 
                Map of token index (in the text) to its obtained embedding
        """
        # groups of well-separated words, represented by their indexes:
        groups = self._group_indexes_separated(ixs=ixs_words)
        map_itok_embeddings = {}
        for group in groups:
            map_itok_embeds = self._get_embeddings_of_sparse_words(
                tokens=tokens,
                ixs_words=group, 
                map_words_to_tokens=map_words_to_tokens)
            map_itok_embeddings = {**map_itok_embeddings, **map_itok_embeds}
        return map_itok_embeddings


    def _get_embeddings_of_sparse_words(self, tokens, ixs_words, map_words_to_tokens):
        """Gets results for the get_embeddings function, which combines the results
        from tokens from groups of sparsely spread words.
        Here the result is obtained for one group of sparsely separated words.
        Args:
            tokens (List[str]): List of tokens, as strings. Represents the text.
            ixs_words (List[int]): List of indexes of words to check
            map_words_to_tokens Dict[int:(int,int)]: 
                Maps each index of word in the text to its range of tokens, the 
                range is: index of first token, index of one-past-last token.
        Returns:
            map_itok_embeds (Dict{int: ndarray[float]}): Map of token index (in text) 
                to its obtained embedding
        """
        map_itok_embeddings = {}
        toks_mask = [map_words_to_tokens[i] for i in ixs_words]  # indexes (beg,end) of tokens to mask
        while toks_mask:
            i_tok_first = toks_mask[0][0]  # first token allowed to mask
            ix_toks_input_beg = max(0, i_tok_first - self.margin)  # input starts here
            ix_toks_input_end = min(len(tokens), ix_toks_input_beg + self.input_size_max)  # input ends here
            i_tok_allowed_last = ix_toks_input_beg + self.input_size_max - self.margin  # last token allowed to mask
            toks_mask_input = []  # tokens to be masked in the input
            for word_toks in toks_mask:
                if word_toks[0] >= i_tok_first and word_toks[1]-1 <= i_tok_allowed_last:
                    toks_mask_input.append(word_toks)
                if word_toks[0] > i_tok_allowed_last:
                    break
            # for preparing next input:
            n_words_used = len(toks_mask_input)
            toks_mask = toks_mask[n_words_used:]
            # get embeddings for the input:
            map_itok_embeds = self._get_embeddings_from_input(
                tokens, 
                ix_toks_input_beg=ix_toks_input_beg, 
                ix_toks_input_end=ix_toks_input_end, 
                toks_mask_input=toks_mask_input)
            map_itok_embeddings = {**map_itok_embeddings, **map_itok_embeds}  # from all inputs so far 
        return map_itok_embeddings


    def _get_embeddings_from_input(self, tokens, ix_toks_input_beg, ix_toks_input_end, toks_mask_input):
        """
        Gets embeddings for one specific input window.
        Returns embeddings for first tokens of each word of interest, while all
        tokens of the word are masked.
        Args:
            tokens (List[str]): Tokens of a summary or text
            ix_toks_input_beg (int): Index of first token of the input window
            ix_toks_input_end (int): Index of the end of the input window
            toks_mask_input (List[(int,int)]): Indexes of all tokens to mask 
                in the input window, given as a list if duples, each duple
                is index of first and one-past last tokens to mask.
        Returns:
            map_itok_embeds (Dict{int: ndarray[float]}): Map of token indexes
                as in text tokens to their embeddings. Covers first tokens
                of all words of interest.
        """
        # Ids of tokens in input for taking embedding
        input_toks = copy.deepcopy(tokens[ix_toks_input_beg:ix_toks_input_end])
        map_itok_iembed = {}
        # Do masking, and also keep record of where to take embeddings from:
        for word_toks in toks_mask_input:
            i_tok_first = word_toks[0]  # first token of the word
            map_itok_iembed[i_tok_first] = 1 + i_tok_first - ix_toks_input_beg  # shift=1 by first [CLS]
            for i in range(word_toks[0], word_toks[1]):
                input_toks[i - ix_toks_input_beg] = '[MASK]'
        # Get embeddings of interest for this input:
        toks_ids = self.model_tokenizer.convert_tokens_to_ids(['[CLS]'] + input_toks + ['[SEP]'])
        input_tensor = torch.LongTensor([toks_ids]).to(self.device)
        outputs = self.model(input_tensor)
        # Contextual embedding:
        emb_all = outputs[1][self.i_layer_context][0]  # all embeddings (for all tokens, at this layer)
        map_itok_embed = {}
        for itok, iembed in map_itok_iembed.items():  # itok is id of token exactly as in tokens
            map_itok_embed[itok] = emb_all[iembed].cpu().detach().numpy()
        return map_itok_embed


    def _evaluate(self, embs_summ, embs_text, embs_summ_raw, embs_text_raw, summ_itokscheck=None):
        """
        Args:
            embs_summ (Dict{int: List[float]}): Map of token indexes 
                (in summary) to the corresponding embeddings
            embs_text (Dict{int: List[float]}): Map of token indexes 
                (in text) to the corresponding embeddings
            embs_summ_raw and embs_text_raw - the same as above, but with the raw embeddings
            summ_itokscheck (Set[int]): Set of indexes of tokens (in summary)
                that must be verified for alarms. 
                This is needed for calculating the original and 'adjusted' ESTIME.
        Returns:
            List[float)] - List of results in order as specified in self.output
        """
        # estime standard, adjusted, all-tokens and soft:
        n_alarms, n_alarms_alltoks, cos_raw_avg = 0, 0, 0
        itoks_similar = []
        for itok_summ, emb_summ in embs_summ.items():   
            tok_summ = self.summ_toks[itok_summ]
            itok_text_best = -1
            sim_best = -1.0e30
            for itok_text, emb_text in embs_text.items():
                sim = dot(emb_summ, emb_text)
                if sim > sim_best:
                    sim_best = sim
                    itok_text_best = itok_text
            tok_text_best = self.text_toks[itok_text_best]
            itoks_similar.append((itok_summ, itok_text_best, sim_best))
            if tok_text_best != tok_summ:
                n_alarms_alltoks += 1
                if not summ_itokscheck or itok_summ in summ_itokscheck:
                    n_alarms += 1
            # Soft estime:
            if self.get_estime_soft:
                emb_summ_nomask = embs_summ_raw[itok_summ]
                emb_text_nomask = embs_text_raw[itok_text_best]
                prod = dot(emb_summ_nomask, emb_text_nomask)
                norm_summ, norm_text = norm(emb_summ_nomask), norm(emb_text_nomask)
                cos_raw_avg += prod / (norm_summ * norm_text)
        if self.get_estime_soft:
            cos_raw_avg /= len(embs_summ)
        # estime-alarms-adjusted:
        if self.get_estime_adjusted:
            if not summ_itokscheck:
                n_alarms_adj = len(embs_summ)
            else:
                n_alarms_adj = n_alarms * len(embs_summ) / len(summ_itokscheck)
        # Coherence:
        if self.get_estime_coherence:
            itoks_summ = [a[0] for a in itoks_similar]
            itoks_text = [a[1] for a in itoks_similar]
            coherence = scipy.stats.kendalltau(itoks_summ, itoks_text, variant='c').correlation
        result = []
        for out_name in self.output:
            if out_name == self.ESTIME_ALARMS:
                result.append(n_alarms)
            elif out_name == self.ESTIME_ADJUSTED:
                result.append(n_alarms_adj)
            elif out_name == self.ESTIME_ALLTOKENS:
                result.append(n_alarms_alltoks)
            elif out_name == self.ESTIME_SOFT:
                result.append(cos_raw_avg)
            elif out_name == self.ESTIME_COHERENCE:
                result.append(coherence)
        return result


    def _select_indexes_separated(self, ixs):
        """Given a list of sorted integers, starts with the first and selects next ones
        in such way that the difference between neighbors is not smaller than the given
        value. Meaning: the integers are the indexes of words in a text.
        Args:
            ixs (List[int]): list of indexes
        Returns:
            ixs_select (List[int]): list of well-separated selected indexes
            ixs_remain (List[int]): list of all the remaining indexes
        """
        ixs_remain = []
        ixs_select = []
        ix_prev = -1000000
        for ix in ixs:
            if ix - ix_prev >= self.distance_word_min:
                ixs_select.append(ix)
                ix_prev = ix
            else:
                ixs_remain.append(ix)
        return ixs_select, ixs_remain


    def _group_indexes_separated(self, ixs):
        """Splits a sorted list of indexes (of words in a text) to groups (lists) 
        of indexes, such that indexes in each groups are separated by the given
        minimal distance.
        Args:
            ixs (List[int]): list of indexes
        Returns:
            groups (List[List[int]]): list of lists of indexes. Each list of indexes
                contains well-separated indexes, satisfying the distance_word_min.
        """
        groups = []
        ixs_remain = copy.deepcopy(ixs)
        while ixs_remain:
            ixs_select, ixs_remain = self._select_indexes_separated(ixs_remain)
            groups.append(ixs_select)
        return groups


    def _get_map_words_intext(self, text_tagged):
        """
        Creates dictionary of words in the text, with all occurrences for each word
        Args:
            text_tagged (List[(str,str)]): List of duples, each is word and its 
                part-of-speach tag. The list is result of nltk.pos_tag function.
        Returns:
            map_words_text Dict{str:List[int]}: Dictionary, key is word from the text,
                value is List[int] - list of all word occurrence indexes in the text
        """
        map_words_text = {}
        for i, (w, t) in enumerate(text_tagged):
            if self.tags_check and t not in self.tags_check:
                continue
            if self.tags_exclude and t in self.tags_exclude:
                continue
            if w not in map_words_text:
                map_words_text[w] = [i]
            else:
                map_words_text[w].append(i)
        return map_words_text  


    def _translate_words_to_tokens(self, text_words):
        """Tokenizes text by model tokenizer.
        Keeps map of indexes of words into indexes of tokens.
        Args:
            text_words (List[str]): Text given as a list of words
        Returns:
            text_tokens (List[str]): Text given as list of tokens
            map_iword_itoks (Dict[int:(int,int)]): Dictionary of the same length 
                as text_words. Word index points to duple of token indexes: 
                index of the first token, and index of the end (one-past-last) token.
        """
        text_tokens = []
        map_iword_itoks = {}
        i_tok = 0
        for ix_word, word in enumerate(text_words):
            toks = self.model_tokenizer.tokenize(word)
            text_tokens.extend(toks)
            i_tok_end = i_tok + len(toks)
            map_iword_itoks[ix_word] = (i_tok, i_tok_end)
            i_tok = i_tok_end
        return text_tokens, map_iword_itoks

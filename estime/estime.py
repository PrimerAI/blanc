import unicodedata
import copy
from numpy import dot

import logging
logging.getLogger('transformers').setLevel(level=logging.WARNING)

import nltk
from nltk.tokenize import word_tokenize

import torch
from transformers import (
    BertForMaskedLM, BertTokenizer, 
    RobertaForMaskedLM, RobertaTokenizer,
    AlbertForMaskedLM, AlbertTokenizer
) 


class Estime:
    """Estimator of factual inconsistencies between summaries (or other textual claims)
    and text, accordingly to ESTIME paper.
    Usage is simple: create `Estime`, and use `evaluate_claims`.
    """
    def __init__(
        self,
        path_mdl,
        i_layer_context,
        device='cpu',
        tags_check=None,
        tags_exclude=None,
        input_size_max=450,
        margin=50,
        distance_word_min=8):
        """
        Args:
            path_mdl (str): path of the model to load and use
            i_layer_context (int): index of layer totake contextual embeddings from
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
        """
        self.device = device
        self.tags_check = tags_check
        self.tags_exclude = tags_exclude
        self.i_layer_context=  i_layer_context
        self.input_size_max = input_size_max
        self.margin = margin
        self.distance_word_min = distance_word_min
        self.model_tokenizer = None
        self.model = None
        if path_mdl.find('roberta') >= 0:
            self.model_tokenizer = RobertaTokenizer.from_pretrained(path_mdl)
            self.model = RobertaForMaskedLM.from_pretrained(path_mdl, output_hidden_states = True).to(self.device)
        elif path_mdl.find('albert') >= 0:
            self.model_tokenizer = AlbertTokenizer.from_pretrained(path_mdl)
            self.model = AlbertForMaskedLM.from_pretrained(path_mdl, output_hidden_states = True).to(self.device)
        else:
            self.model_tokenizer = BertTokenizer.from_pretrained(path_mdl)
            self.model = BertForMaskedLM.from_pretrained(path_mdl, output_hidden_states = True).to(self.device)
        self.model.eval()
        # Convenient data (tokenized summary and text):
        self.summ_toks = None 
        self.text_toks = None

    def evaluate_claims(self, text, claims):
        """
        Given a text, and a list of claims (e.g. summaries, or other texts), 
        estimates how many likely factual inconsistencies each claim contains
        (the inconsistencies are with respect to the text).
        Number of alarms is the main result, because it correlates well with 
        human-scored factual consistency. Number of winners is comparable.
        The count of new claim words is just for info.
        Args:
            text (str): the main text
            claims (List[str]): texts ('claims') to be checked for consistency 
                with the main text. Each claim is preferably a shorter text, 
                like a claim/statement/summary.
        Returns: 
            claims_info (List[(int,int,int)]): List of tuples. Each tuple is:
                1. Number of alarms. Each alarm is a location in the claim that 
                    might be inconsistent with respect to the text
                1. Number of winners. Each winner is a probable substitution 
                    from the text into an alarmed location in the claim
                3. Number of claim words that are not in the text
        """
        # Text:
        text = unicodedata.normalize('NFKD', text)
        text_words = word_tokenize(text)
        text_tagged = nltk.pos_tag(text_words)
        # Find all words of interest in the text, tokenize:
        text_map_wordscheck = self._get_map_words_intext(text_tagged)
        text_iwordscheck = sorted([item for sublist in text_map_wordscheck.values() for item in sublist])
        self.text_toks, text_map_iword_itoks = self._translate_words_to_tokens(text_words)
        # All embeddings of interest in the text:
        embs_text = self._get_embeddings(
            tokens=self.text_toks, 
            ixs_words=text_iwordscheck, 
            map_words_to_tokens=text_map_iword_itoks)
        # For each claim, count number of signals for potential inconsistencies:
        claims_info = []
        for claim in claims:
            claim = unicodedata.normalize('NFKD', claim)
            n_alarms, n_winners, n_newwords = self._evaluate_claim(
                claim, text_map_wordscheck, embs_text)
            claims_info.append((n_alarms, n_winners, n_newwords))
        self.summ_toks = None 
        self.text_toks = None
        return claims_info

    def _evaluate_claim(self, claim, words_check, embs_text):
        """Finding number of alarms, winners and new words for a claim.
        Number of alarms is the main result, because it correlates well with 
        human-scored factual consistency. 
        The counts of new words is just for info.
        Text is already processed, its embeddings can be used for each claim.
        Args:
            claim (str): claim, e.g. summary or short text - not the main text
            words_check (Set{str}): a set or map where the keys are the words 
                of interest in the main text
            embs_text (Dict{int: List[float]}): Map of token index (in text) 
                to its obtained embedding
        Returns:
            n_alarms, n_winners, n_newwords (int,int,int): Number of alarms, 
                winners and new words for the claim. Each alarm is a location 
                in the claim that may be inconsistent with respect to the text. 
                Each winner is a plausible substitution from the text into an 
                alarmed location in the claim. Each new word is a word existng 
                in the claim but not in the text.
        """
        summ_words = word_tokenize(claim)
        summ_tagged = nltk.pos_tag(summ_words)
        # Find all words of interest in the summary:
        summ_iwordscheck = []
        summ_wordsnew = []
        for i, (w, t) in enumerate(summ_tagged):
            if w not in words_check:  # checking only what exists in the text
                summ_wordsnew.append(w)
            else:
                summ_iwordscheck.append(i)
        n_newwords = len(summ_wordsnew)
        self.summ_toks, summ_map_iword_itoks = self._translate_words_to_tokens(summ_words)
        embs_summ = self._get_embeddings(
            tokens=self.summ_toks, ixs_words=summ_iwordscheck, map_words_to_tokens=summ_map_iword_itoks)
        # All scalar products:
        map_itokspair_sim = self._get_all_prods(embs_summ=embs_summ, embs_text=embs_text)
        # Find max similarity with all occurrences of the incumbent tokens in the text:
        map_itok_occurrs = self._find_occurrences_of_tokens(embs_summ, embs_text, map_itokspair_sim)
        map_itok_csMax = {}
        for itok, occurrs in map_itok_occurrs.items():
            cs_max = max([a[1] for a in occurrs])
            map_itok_csMax[itok] = cs_max
        map_itok_winners = self._find_candidates(embs_summ, embs_text, map_itokspair_sim, map_itok_csMax)
        n_alarms = sum([1 for k,v in map_itok_winners.items()])
        n_winners = sum([len(v) for k,v in map_itok_winners.items()])
        return n_alarms, n_winners, n_newwords
    
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
        map_itok_embeds = {}
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
            map_itok_embed = self._get_embeddings_from_input(
                tokens, 
                ix_toks_input_beg=ix_toks_input_beg, 
                ix_toks_input_end=ix_toks_input_end, 
                toks_mask_input=toks_mask_input)
            map_itok_embeds = {**map_itok_embeds, **map_itok_embed}  # from all inputs so far
        return map_itok_embeds

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
                as in text 'tokens' to their embeddings. Covers first tokens
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
        emb_all = outputs[1][self.i_layer_context][0]  # all embeddings
        map_itok_embed = {}
        for itok, iembed in map_itok_iembed.items():  # itok is id of token exactly as in tokens
            map_itok_embed[itok] = emb_all[iembed].cpu().detach().numpy()
        return map_itok_embed

    def _find_occurrences_of_tokens(self, embs_summ, embs_text, map_itokspair_sim):
        """
        Finds all occurences of summary tokens in the text. 
        Finds also similarity of each occurrence to the summary token.
        Args:
            embs_summ (Dict{int: List[float]}): Map token index in summary to embedding
            embs_text (Dict{int: List[float]}): Map token index in text to embedding
            map_itokspair_sim (Dict{(int,int)):float}: Dictionary containing all 
                products between summary and text embeddings of interest.
                Key = duple: index of token in summary and index of token in text
                Value = similarity of their embeddings
        Returns:
            map_itok_occurrs (Dict{int:List[(int,float)]}): Maps locations of
                summary tokens to locations of their occurrences in the text.
                key: index of suspect-token in summary; value: list of duples. 
                Each duple: (1) index of an occurrence of the token in the text, 
                (2) similarity of the occurrence to the token in the summary.
        """
        # Find all occurrences of the incumbent token:
        map_itok_occurrs = {}  # info about all occurrences of the token in text
        for itok_summ in embs_summ.keys():  # itok is index of the token in summary
            tok_summ = self.summ_toks[itok_summ]
            occurrs = []
            for itok_text in embs_text.keys():
                tok_text = self.text_toks[itok_text]
                if tok_summ != tok_text:
                    continue
                sim = map_itokspair_sim[(itok_summ, itok_text)]
                occurrs.append((itok_text, sim))
            assert occurrs  # Checking only tokens existing in the text
            map_itok_occurrs[itok_summ] = copy.deepcopy(occurrs)
        return map_itok_occurrs

    def _find_candidates(self, embs_summ, embs_text, map_itokspair_sim, map_itok_csMax):
        """
        For each summary token finds all text tokens that could replace it
        because they have higher context similarity to it. Returns info about
        all summary tokens for which such potential replacements were found.
        Tokens here are by the model's tokenizer.
        Args:
            embs_summ (Dict{int: List[float]}): Map of token indexes 
                (in summary) to the corresponding embeddings
            embs_text (Dict{int: List[float]}): Map of token indexes 
                (in text) to the corresponding embeddings
            map_itokspair_sim (Dict{(int,int)):float}: Dictionary containing all 
                products between summary and text embeddings of interest.
                Key = duple: index of token in summary and index of token in text
                Value = similarity of their embeddings
            map_itok_csMax (Dict{int:float}): Map of token index in summary 
                to maximal similarity of its embedding to embeddings of all 
                occurrences of this token in the text.
        Returns:
            map_itok_winners Dict{int: List[(int,float)]}: A map giving info 
                about all suspicious tokens in the summary.
                A suspicious token is a summary token, for which the text has 
                at least one token that is different as a string, but has 
                embedding more similar to the summary tokem's embedding than 
                all embedding of the occurrences of summary token in the text.
                Key: index of a suspect token in the summary
                Value: List of duple, listing all the 'winners' in the text. 
                    Each duple is: index of a winner-token in the text, and
                    similarity of its embedding to summary-token embedding.
        """
        # Find all candidates that are different from the incumbent:
        map_itok_winners = {}
        for itok_summ in embs_summ.keys():
            tok_summ = self.summ_toks[itok_summ]
            sim_threshold = map_itok_csMax[itok_summ]  # must get better than this
            winners = []  # All winning candidates for this token
            for itok_text in embs_text.keys():
                tok_text = self.text_toks[itok_text]
                if tok_summ == tok_text:
                    continue  # not interesting in occurrences of incumbent token
                sim = map_itokspair_sim[(itok_summ, itok_text)]
                if sim > sim_threshold:
                    winners.append((itok_text, sim))
            if winners:
                map_itok_winners[itok_summ] = copy.deepcopy(winners)
        return map_itok_winners

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

    def _get_all_prods(self, embs_summ, embs_text):
        """Finds products of all embeddings of interest.
        Given lists of embeddings (and their indexes) from summary and from text,
        returns their products.
        """
        map_itokspair_sim = {}
        for itok_summ, emb_summ in embs_summ.items():  # itok is token index in summary
            for itok_text, emb_text in embs_text.items():
                sim = dot(emb_summ, emb_text)
                map_itokspair_sim[(itok_summ, itok_text)] = float(sim)
        return map_itokspair_sim
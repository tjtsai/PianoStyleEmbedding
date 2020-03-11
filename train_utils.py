#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import torch
from fastai import *
from fastai.text import *
from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import SequenceSummary

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, tok_dir, max_seq_len, **kwargs):
        
        tokenizer = CharBPETokenizer(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt")
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=max_seq_len)

        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the special tokens"""
        output = self._pretrained_tokenizer.encode(t)
        return output.tokens

class CustomTokenizer():
    '''Wrapper for TransformersBaseTokenizer tokenizer to fit into fast.ai'''
    def __init__(self,tok_func:Callable, model_prefix:str, max_seq_len:int):
        self.tok_func, self.model_prefix, self.max_seq_len = tok_func, model_prefix, max_seq_len
        
    def __repr__(self) -> str:
        res = f'Tokenizer {self.tok_func.__name__} using `{self.model_prefix}` model\n'
        return res

    def process_text(self, t:str, tok:BaseTokenizer) -> List[str]:
        "Process one text `t` with tokenizer `tok`."
        toks = tok.tokenizer(t)
        return toks 
    
    def _process_all_1(self,texts:Collection[str]) -> List[List[str]]:
        'Process a list of `texts` in one process'
        tok = self.tok_func(self.model_prefix, self.max_seq_len)
        return [self.process_text(t, tok) for t in texts]
                                                                     
    def process_all(self, texts:Collection[str]) -> List[List[str]]: 
        "Process a list of `texts`."                                 
        return self._process_all_1(texts)

class TransformersVocab(Vocab):
    """ Subclassing the fastai Vocab class to use the custom tokenizer. """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return [self.tokenizer.token_to_id(tok) for tok in t]

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        strings = [self.tokenizer.id_to_token(num) for num in nums]
        if sep is None:
            sep = ''
        return sep.join(strings)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

class RobertaModelWrapper(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel, pad_idx:int):
        super(RobertaModelWrapper,self).__init__()
        self.transformer = transformer_model
        self.pad_idx = pad_idx
        
    def forward(self, input_ids, attention_mask=None):
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=self.pad_idx).type(input_ids.type()) 
        logits = self.transformer(input_ids, attention_mask = attention_mask)[0]   
        return logits
    
    def reset(self): pass

class GPT2ModelWrapper(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel, pad_idx:int, cls_idx:int):
        super(GPT2ModelWrapper,self).__init__()
        self.model = transformer_model
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx
        
    def forward(self, input_ids, attention_mask=None):
        mc_token_ids = (input_ids==self.cls_idx).nonzero()[:,1]
        assert mc_token_ids.shape[0] == input_ids.shape[0]        
        outputs = self.model(input_ids, mc_token_ids=mc_token_ids, attention_mask=attention_mask)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]
        return mc_prediction_scores
    
    def reset(self): pass

class GPT2Classifier(nn.Module):
    def __init__(self, transformer_model:PreTrainedModel, config:PretrainedConfig, pad_idx:int, cls_idx:int):
        super(GPT2Classifier,self).__init__()
        self.transformer = transformer_model
        self.head = SequenceSummary(config)
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        mc_token_ids = (input_ids==self.cls_idx).nonzero()[:,1]
        assert mc_token_ids.shape[0] == input_ids.shape[0]
        hidden_states = transformer_outputs[0]
        out = self.head(hidden_states, mc_token_ids).squeeze(-1)
        return out
    
def plotLosses(logfile):
    
    # parse log file
    tr_loss = []
    val_loss = []
    step = []
    with open(logfile, 'r') as f:
        for line in f:
            m = re.search('Step (\d+): train loss = (\S+), valid loss = (\S+),', line.strip())
            if m:
                step.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
    
    # plot
    step, tr_loss, val_loss = np.array(step), np.array(tr_loss), np.array(val_loss)
    plt.plot(step, tr_loss, 'k-')
    plt.scatter(step, tr_loss, c='k')
    plt.plot(step, val_loss, 'g-')
    plt.scatter(step, val_loss, c='g')
    plt.legend(['train','valid'])
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()

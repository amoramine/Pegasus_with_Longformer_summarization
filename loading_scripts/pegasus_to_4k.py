import argparse
import logging
import os
import copy
import torch #ADDED
import tensorflow as tf

#from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart
#from longformer.longformer_encoder_decoder import LongformerEncoderDecoderConfig
#from longformer.longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration

from longformer_for_pegasus import LongformerSelfAttentionForPegasus
from longformer_for_pegasus import LongformerEncoderDecoderConfig
from longformer_for_pegasus import LongformerEncoderDecoderForConditionalGeneration


from transformers.modeling_longformer import LongformerSelfAttention 
from transformers import PegasusTokenizer
from transformers import PegasusForConditionalGeneration

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def create_long_model(save_model_to, attention_window, max_pos):
    # base_model = 'google/pegasus-xsum'
    # base_model = 'sshleifer/student-pegasus-xsum-6-6'
    base_model = 'sshleifer/distill-pegasus-cnn-16-4'
    model = PegasusForConditionalGeneration.from_pretrained(base_model)

    tokenizer = PegasusTokenizer.from_pretrained(base_model, model_max_length=max_pos)
    config = LongformerEncoderDecoderConfig.from_pretrained(base_model)
    
    

    model.config = config

    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = ['LongformerEncoderDecoderForConditionalGeneration', ]

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    assert current_max_pos == config.max_position_embeddings

    config.max_encoder_position_embeddings = max_pos 
    config.max_decoder_position_embeddings = config.max_position_embeddings #512
    print("max_encoder_position_embeddings: ", config.max_encoder_position_embeddings)
    print("max_decoder_position_embeddings: ", config.max_decoder_position_embeddings)

    del config.max_position_embeddings

    assert max_pos > current_max_pos


    # allocate a larger position embedding matrix
    new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings

    # k = 0
    # step = current_max_pos
    k = 0
    step = current_max_pos - k
    while k < max_pos - 1:
        new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[:]
        k += step

    model.model.encoder.embed_positions = torch.nn.Embedding.from_pretrained(new_encoder_pos_embed)

    print(model.model.encoder.layers)


    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_pegasus = LongformerSelfAttentionForPegasus(config, layer_id=i)

        longformer_self_attn_for_pegasus.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_pegasus.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_pegasus.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_pegasus.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
        longformer_self_attn_for_pegasus.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
        longformer_self_attn_for_pegasus.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

        longformer_self_attn_for_pegasus.output = layer.self_attn.out_proj
        layer.self_attn = longformer_self_attn_for_pegasus

    print("OK")

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def main():
    model, tokenizer = create_long_model(save_model_to="Pegasus_4k/", attention_window=512, max_pos=4096)


if __name__ == "__main__":
    main()


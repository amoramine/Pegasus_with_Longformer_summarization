# Pegasus_with_Longformer_summarization

## Description

Pegasus is a large Transformer-based encoder-decoder model with a new pre-training objective which is adapted to abstractive summarization. More specifically, the pre-training objective, called "Gap Sentence Generation (GSG)", consists of masking important sentences from a document and generating these gap-sentences.

On the other hand, the Longformer is a Transformer which replaces the full-attention mechanism (quadratic dependency) with a novel attention mechanism which scale linearly with the input sequence length. Consequently, Longformer can process sequences up to 4,096 tokens long (8 times longer than BERT which is limited to 512 tokens).

This project plugs Longformer's attention mechanism to Pegasus in order to perform abstractive summarization on long documents. The conversion is done in loading_scripts/Pegasus_to_4k.py which enables Pegasus to process sequences up to 4,096 tokens long (rather than 512 tokens). Note that the `max_pos` parameter can be changed to accept even longer sequences (e.g `max_pos=16384`). The new Pegasus model is then fine-tuned on BigPatent dataset. To assess the model's performance on long documents, all training examples are filtered such that they have a minimum length of 4000 tokens.
 
This project was built using HuggingFace's Transformers library. The model is trained using model partitioning (with fairscale) and parallel batch processing on a cluster of 8 GPUs.

## How to run the project
To run this project, clone the repo and execute the following commands:

1) `cd Pegasus_with_Longformer_summarization`
2) `pip install -r requirements.txt`
3) `pip install git+https://github.com/allenai/longformer.git`
4) `pip install tokenizers==0.10.3`
5) Comment out `import 'SAVE_STATE_WARNING' from torch.optim.lr_scheduler` in lib/python3.7/site-packages/transformers/trainer_pt_utils.py
6) Add `with torch.no_grad():` above `out[:, 0 : dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))` in lib/python3.7/site-packages/transformers/modeling_bart.py
7) `python loading_scripts/pegasus_to_4k.py`
8) `git clone -b v4.5.1-release https://github.com/huggingface/transformers`
9) `cd transformers`
10) `pip install -e .` 
11) `cd .. ; python download_long_Big_Patent_data.py` 
12) `bash tune.sh`

## Citation

@article{DBLP:journals/corr/abs-1910-03771,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  author    = {Thomas Wolf and
               Lysandre Debut and
               Victor Sanh and
               Julien Chaumond and
               Clement Delangue and
               Anthony Moi and
               Pierric Cistac and
               Tim Rault and
               Rémi Louf and
               Morgan Funtowicz and
               Jamie Brew},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  title     = {HuggingFace's Transformers: State-of-the-art Natural Language Processing},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  journal   = {CoRR},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  volume    = {abs/1910.03771},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  year      = {2019},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  url       = {http://arxiv.org/abs/1910.03771},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  archivePrefix = {arXiv},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  eprint    = {1910.03771},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  timestamp = {Tue, 02 Jun 2020 12:49:01 +0200},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  biburl    = {https://dblp.org/rec/journals/corr/abs-1910-03771.bib},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  bibsource = {dblp computer science bibliography, https://dblp.org}<br/>
}

@article{DBLP:journals/corr/abs-2004-05150,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  author    = {Iz Beltagy and
               Matthew E. Peters and
               Arman Cohan},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  title     = {Longformer: The Long-Document Transformer},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  journal   = {CoRR},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  volume    = {abs/2004.05150},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  year      = {2020},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  url       = {https://arxiv.org/abs/2004.05150},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  archivePrefix = {arXiv},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  eprint    = {2004.05150},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  timestamp = {Tue, 14 Apr 2020 16:40:34 +0200},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-05150.bib},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  bibsource = {dblp computer science bibliography, https://dblp.org}<br/>
}

@article{DBLP:journals/corr/abs-1912-08777,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  author    = {Jingqing Zhang and
               Yao Zhao and
               Mohammad Saleh and
               Peter J. Liu},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  title     = {{PEGASUS:} Pre-training with Extracted Gap-sentences for Abstractive
               Summarization},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  journal   = {CoRR},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  volume    = {abs/1912.08777},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  year      = {2019},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  url       = {http://arxiv.org/abs/1912.08777},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  archivePrefix = {arXiv},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  eprint    = {1912.08777},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  timestamp = {Fri, 03 Jan 2020 16:10:45 +0100},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  biburl    = {https://dblp.org/rec/journals/corr/abs-1912-08777.bib},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;  bibsource = {dblp computer science bibliography, https://dblp.org}<br/>
}

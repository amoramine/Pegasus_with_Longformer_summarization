# pegasus_longformer

To run this project, clone the repo and run the following commands:

1) cd pegasus_longformer
2) pip install -r requirements.txt
3) pip install git+https://github.com/allenai/longformer.git
4) pip install --upgrade tokenizers
5) comment out "import 'SAVE_STATE_WARNING' from torch.optim.lr_scheduler" in lib/python3.7/site-packages/transformers/trainer_pt_utils.py
6) python loading_scripts/pegasus_to_4k.py
7) git clone -b v4.5.1-release https://github.com/huggingface/transformers
8) cd transformers
9) pip install -e . 
10) bash tune.sh

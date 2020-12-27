# eece695h_Fewshot_RN
Pytorch implementation of the 5-way, 5-shot classification task over CUB dataset. Implementation based on paper "Learning to compare: relation network for few shot learning", CVPR 2018.


## Train
To train the model, run main.py:

    python main.py

## Test
To test the model, run main.py with appropriate checkpoints for both embedding and relation module.

    python main.py --load_emb_ckpt ./checkpoints/embed_module_checkpoint.pth --load_rel_ckpt ./checkpoints/relation_module_checkpoint.pth --test_mode 1
 

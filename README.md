# eece695h_Fewshot_RN
Pytorch implementation of the 5-way, 5-shot classification task over CUB dataset. 
Implementation is based on paper "Learning to compare: relation network for few shot learning", CVPR 2018.


## Train
To train the model, run main.py:

    python main.py

## Test
To test the model, run main.py with appropriate checkpoints for both embedding and relation module.

    python main.py --load_emb_ckpt ./checkpoints/embed_module_checkpoint.pth --load_rel_ckpt ./checkpoints/relation_module_checkpoint.pth --test_mode 1
 
## Dataset
Download the CUB-200-2011 dataset (All Images and Annotations) from the official website.
(http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
Then save it to './dataset/' path.

## Visualization 
If you want to visualize your training loss, accuracy and validation loss, accuracy, 
then install tensorboardx library for pytorch. 

    pip install tensorboardx
    
Then, in another terminal command line, run the below:

    tensorboard --logdir=runs/your_run_file
    ex) tensorboard --logdir=runs/Dec24_21-30-44_gpu03
    
    

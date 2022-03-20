<h3 align="center">
    Bilateral Personalized Dialogue Generation with Contrastive Learning
</h3>
<h4 align="center">
    基于对比学习的双边个性化对话生成
</h4>
<hr>


<h3 align="center">
    
</h3>


#####        
Generating personalized responses is one of the
major challenges in natural human-robot interaction. Current
studies in this field mainly focus on generating responses
consistent with the robot's pre-assigned persona, while ignoring the user's persona. Such responses may be inappropriate or even offensive, which may lead to the bad user experience.
Therefore, we propose a Bilateral Personalized Dialogue Generation (BPDG) method for dyadic conversation, which integrates user and robot personas into dialogue generation via designing a dynamic persona-aware fusion method. 
To bridge the gap between the learning objective function and evaluation metrics, the Conditional Mutual Information Maximum (CMIM) criterion is adopted with contrastive learning to select the proper response from the generated candidates. Moreover, a bilateral persona accuracy metric is designed to measure the degree of bilateral personalization.  
Experimental results demonstrate that, compared with several state-of-the-art methods, the proposed method achieves the improvement in the 23.99 of bilateral persona accuracy, 1.1 in BLEU, 0.83 in F1, 0.02 in distinct score on the random personalized test set, and the improvement in 5.56 of bilateral persona accuracy, 7.51 in BLEU, 2.12 in F1, 0.02 in distinct score on the biased personalized test set.
On the manual evaluations, the proposed method can generate more fluency, bilateral persona-consistent, and context-consistent responses compared with other state-of-the-art methods.

## Preliminary

Please download the pre-trained model from [Github Link](https://github.com/thu-coai/CDial-GPT) 

The model and experimental records after training will be saved in the `/parameters/` folder.

For training dataset, please refer to the paper [AAAI paper](https://arxiv.org/abs/1911.04700) 

For testing dataset, please refer to the link [website Link](https://worksheets.codalab.org/worksheets/0x8f68b61a8b2249d7b314c6e800e2dace) 

# BPDG
This code is now open-sourced, but not well cleaned. This issues will be fixed in the future.

## For Our BPDG model

### For Training

1. python train_our_v3.py

### For testing

2. python generate_our_sample.py

This repository  also contains other state-of-the-art methods for reproductions.

If you have any questions, feel free to email me at libincn@hnu.edu.cn.

## Citation
Please feel free to cite our [paper]{https://arxiv.org/abs/2106.07857).

    @article{li2021bilateral,
      title={Bilateral Personalized Dialogue Generation with Contrastive Learning},
      author={Li, Bin and Deng, Hanjun},
      journal={arXiv preprint arXiv:2106.07857},
      booktitle={Arxiv},
      year={2021},
      url={https://arxiv.org/abs/2106.07857}
    }



# Bristol-Myers-Squibb-Molecular-Translation

This respository contains my code for competition in kaggle.

7th Place Solution for [Bristol-Myers Squibb – Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation)


Team: [Mr_KnowNothing](https://www.kaggle.com/tanulsingh077), [Shivam Gupta](https://www.kaggle.com/shivamcyborg), [Phaedrus](https://www.kaggle.com/pheadrus), [Nischay Dhankhar](https://www.kaggle.com/nischaydnk), [atfujita](https://www.kaggle.com/atsunorifujita)

- All models(Team)    
Public LB: 0.60(7th)    
Private LB: 0.60(7th)    

The full picture of our solution is [here](https://www.kaggle.com/c/bms-molecular-translation/discussion/243779)

_**Note: This repository contains only my models and only train script.**_    

- My models(3 Models averaging)    
Public LB: 0.66    
Private LB: 0.66    

_**Note: This repogitory is based on [hengck23](https://www.kaggle.com/hengck23)'s great assets. Please check [here](https://www.kaggle.com/c/bms-molecular-translation/discussion/231190) for details**_

## My Models

Vit based model
- Encoder: vit_deit_base_distilled_patch16_384
- Decoder: TransformerDecoder
- Loss: LabelSmoothingLoss
- Augumentation: RandomScale, Cutout

There are 2 Vit based models.    
The second was re-training by strengthening Noize Injection and Augmentation.

With Normalize    
Public LB: 0.77    
Private LB: 0.78    

With Normalize    
Public LB: 0.76    
Private LB: 0.77    


Swin Transformer based model
- Encoder: swin_base_patch4_window12_384_in22k
- Decoder: TransformerDecoder
- Loss: LabelSmoothingLoss
- Augumentation: RandomScale, Cutout

With Normalize    
Public LB: 0.91    
Private LB: 0.92    

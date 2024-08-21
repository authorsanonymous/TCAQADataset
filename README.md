## About the CoSL model：

### Requirement
Python >= 3.6

Pytorch >=1.8.0

### Dataset Preparation
**1.TCAQA dataset**

If the article is accepted for publication, you can download our prepared TCAQA dataset demo from ["Google Drive"](https://drive.google.com/file/d/13Rr3XIo5t2QygmerOVCFn1pRiyg4wPVC/view?usp=sharing) . Then, please move the uncompressed data folder to `TCAQA/data/frames`. We used the I3D backbone pretrained on Kinetics([Google Drive](https://drive.google.com/file/d/1M_4hN-beZpa-eiYCvIE7hsORjF18LEYU/)).

**2.MTL-AQA dataset**(["Google Drive"](https://drive.google.com/file/d/1T7bVrqdElRLoR3l6TxddFQNPAUIgAJL7/))

### Training & Evaluation
In this paper, we selected classic action quality assessment models such as ResNet-WD, USDL/MUSDL, and DAE as example models, and then integrated CoSL into each of these models, namely ResNet-WD-CoSL, USDL-CoSL, DAE-CoSL, and MUSDL-CoSL, aiming to improve the overall performance of the network by learning constraint information from score labels. Here,take **MUSDL-CoVL** as an example,To train and evaluate on TCAQA:

` python -u main.py  --lr 1e-4 --weight_decay 1e-5 --gpu 0 `

If you use the TCAQA dataset, please cite this paper: A Teacher Action Quality Assessment Method Based on Contrastive Score Label.

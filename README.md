This is a source code for the manuscript titled "SynerMix: Synergistic Mixup Solution for Enhanced Intra-Class Cohesion and Inter-Class Separability in Image Classification". The manuscript has been submitted to ArXiv https://arxiv.org/abs/2403.14137.  
![GRAPH_ABSTRACT](https://github.com/wxitxy/synermix/assets/129836406/19ae0685-5a2f-46fc-b6ff-7b7874309f60)

  
To use the code, follow these steps:  
1. Install the required dependencies by running `pip install -r requirements.txt`.  
2. Execute `main.py`. The results and model files will be stored in the respective `logs` and `models` folders. The six methods mentioned in the paper, namely wo-RA&ER, w-RA, w-ER(M), w-ER(MM), w-RA&ER(M), and w-RA&ER(MM), can be run as follows, using the CIFAR-100 dataset as an example:  
   - wo-RA&ER: `python main.py -mn resnet-18 -ds cifar100 -bs 128`  
   - w-RA: `python main.py -mn resnet18 -ds cifar100 -bs 128 -beta 0.5 -phase test`  
   - w-ER(M): `python main.py -mn resnet18 -ds cifar100 -bs 128 -alpha 1 -itrm Mixup -phase test`  
   - w-ER(MM): `python main.py -mn resnet18 -ds cifar100 -bs 128 -alpha 1 -itrm Manifold_Mixup -phase test`  
   - w-RA&ER(M): `python main.py -mn resnet18 -ds cifar100 -bs 128 -alpha 1 -beta 0.1 -itrm Mixup -phase test`  
   - w-RA&ER(MM): `python main.py -mn resnet18 -ds cifar100 -bs 128 -alpha 1 -beta 0.1 -itrm Manifold_Mixup -phase test`  

To enable mixed precision training for reduced computational costs, append `-mp true` to the command, for example:  
`python main.py -mn resnet18 -ds cifar100 -bs 128 -beta 0.5 -mp true`  
  
Parameter details:  
1. -mn: Model Name (default: resnet18, options: resnet18, resnet34, resnet50, resnet101, mobilenet, tiny-swin)  
2. -ds: Dataset Name (default: cifar100, options: food101, miniimagenet, oxfordiiipet, caltech256)  
3. -pr: Using a Pre-trained Model or Not (default: False)  
4. -bs: Batch Size (default: 128)  
5. -alpha: Alpha (default: 1)  
6. -beta: Beta (default: 0)  
7. -itrm: Mixup Method (default: None, options: Mixup, Manifold_Mixup)  
8. -lr: Learning Rate (default: 0.1)  
9. -mo: Momentum (default: 0.9)  
10. -wd: Weight Decay (default: 5e-4)  
11. -ss: Step Size for Learning Rate Decay (default: 10)  
12. -ga: Gamma for Learning Rate Decay (default: 0.5)  
13. -seed: Random Seed (default: 123)  
14. -ep: Total Number of Epochs (default: 120)  
15. -ne: Using Nesterov or Not (default: False)  
16. -phase: Testing or Validation (default: test, options: test, val)  
17. -vr: Validation Ratio (default: 0.1)  
18. -mp: Using Mixed Precision Training or Not (default: False)  

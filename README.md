这是一份针对手稿名为"Improving Image Classification Accuracy through Intra-Class Mixup and Inter-Class Mixup"的源代码。手稿已被提交到Neurocomputing期刊。  
  
代码使用步骤如下：  
1. 通过pip install -r requirements.txt安装依赖环境。  
2. 执行main.py,运行结果和模型文件将分别被放在生成的logs文件夹和models文件夹中。论文中涉及的六种方法wo-RA&ER, w-RA, w-ER(M), w-ER(MM), w-RA&ER(M), w-RA&ER(MM)的运行方式如下, 以cifar-100数据集为例：  
    wo-RA&ER: python main.py -mn resnet-18 -ds cifar100 -bs 128  
    w-RA: python main.py -mn resnet18 -ds cifar100 -bs 128 -beta 0.5 -phase test  
    w-ER(M): python main.py -mn resnet18 -ds cifar100 -bs 128 -alpha 1 -itrm Mixup -phase test  
    w-ER(MM): python main.py -mn resnet18 -ds cifar100 -bs 128 -alpha 1 -itrm Manifold_Mixup -phase test  
    w-RA&ER(M)：python main.py -mn resnet18 -ds cifar100 -bs 128 -alpha 1 -beta 0.1 -itrm Mixup -phase test  
    w-RA&ER(MM)：python main.py -mn resnet18 -ds cifar100 -bs 128 -alpha 1 -beta 0.1 -itrm Manifold_Mixup -phase test  

代码通过追加-mp true实现混合精度训练，以减少计算成本，如:  
python main.py -mn resnet18 -ds cifar100 -bs 128 -beta 0.5 -mp True  
  
参数详情：  
1. -mn           Model Name (default: resnet18, options：resnet18, resnet34, resnet50, resnet101, mobilenet, tiny-swin)  
2. -ds           Dataset Name (default：cifar100, options: food101, miniimagenet, oxfordiiipet, caltech256)  
3. -pr           Using a Pre-trained Model or Not (default: False)  
4. -bs           Batch Size (default: 128)  
5. -alpha        Alpha (default: 1)  
6. -beta         Beta (default: 0)  
7. -itrm         Mixup Method (default: None，options: Mixup, Manifold_Mixup)  
8. -lr           Learning Rate (default: 0.1)  
9. -mo           Momentum (default: 0.9)  
10. -wd          Weight Decay (default: 5e-4)  
11. -ss          Step Size for Learning Rate Decay (default: 10)  
12. -ga          Gamma for Learning Rate Decay (default: 0.5)  
13. -seed        Random Seed (default: 123)  
14. -ep          Total Number of Epochs (default: 120)  
15. -ne          Using Nesterov or Not (default: False)  
16. -phase       Testing or Validation (default：test, options: test, val)  
17. -vr          Validation Ratio (default：0.1)  
18. -mp          Using Mixed Precision Training or Not (default：False)  

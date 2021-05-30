# Awesome deep learning papers
### Must-read papers and must-known concepts
<br/>

## [Architectures](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/Architectures.rst)
Neural networks evolution
<br/>
<br/>
[Review: Inception-v3 Sep 10 2018](https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)
<br/>
<br/>
[Dilated Residual Networks 28 May 2017](https://arxiv.org/pdf/1705.09914.pdf)
<br/>
<br/>
[Attention Is All You Need [6 Dec 2017]](https://arxiv.org/pdf/1706.03762.pdf)
(https://habr.com/ru/post/486158/)
<br/>
<br/>
[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks 28 May 2019](https://arxiv.org/abs/1905.11946)
<br/>
<br/>
<br/>
## [Training](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/Training.rst)
Optimization techniques, regularization, tips & tricks, distribution
<br/>
<br/>
<br/>
[A Recipe for Training Neural Networks Apr 25 2019](http://karpathy.github.io/2019/04/25/recipe/)
<br/>
<br/>
[Multi-GPU Training of ConvNets 18 Feb 2014](https://arxiv.org/pdf/1312.5853.pdf)
<br/>
<br/>
[The Effectiveness of Data Augmentation in Image Classification using Deep Learning 13 Dec 2017](https://arxiv.org/pdf/1712.04621.pdf)
<br/>
<br/>
[Parallel and Distributed Deep Learning](https://web.stanford.edu/~rezab/classes/cme323/S16/projects_reports/hedge_usmani.pdf)
<br/>
<br/>
[Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis 15 Sep 2018](https://arxiv.org/pdf/1802.09941.pdf)
<br/>
<br/>
[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour 30 Apr 2018](https://arxiv.org/pdf/1706.02677.pdf)
<br/>
<br/>
[Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes 30 Jul 2018](https://arxiv.org/pdf/1807.11205.pdf)
<br/>
<br/>
[MIXED PRECISION TRAINING 15 Feb 2018](https://arxiv.org/pdf/1710.03740.pdf)
<br/>
<br/>
[A Survey on Distributed Machine Learning 20 Dec 2019](https://arxiv.org/ftp/arxiv/papers/1912/1912.09789.pdf)
<br/>
<br/>
[NVIDIA Deep Learning Performance](https://docs.nvidia.com/deeplearning/performance/index.html)
<br/>
<br/>
[Stochastic Weight Averaging](https://towardsdatascience.com/stochastic-weight-averaging-a-new-way-to-get-state-of-the-art-results-in-deep-learning-c639ccf36a)
<br/>
<br/>
[Bag of Tricks for Image Classification with Convolutional Neural Networks 5 Dec 2018](https://arxiv.org/pdf/1812.01187.pdf)
<br/>
<br/>
[Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)
<br/>
<br/>
[Training Generative Adversarial Networks with Limited Data 7 Oct 2020](https://arxiv.org/pdf/2006.06676.pdf)
<br/>
<br/>
<br/>
## [Theory](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/Theory.rst)
 Understanding how neural networks work on a deeper level
<br/>
<br/>
<br/>
[Deconvolution and Checkerboard Artifacts 2016](http://doi.org/10.23915/distill.00003)
<br/>
<br/>
[UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION 26 Feb 2017](https://arxiv.org/pdf/1611.03530.pdf)
<br/>
<br/>
[Emergence of Invariance and Disentanglement in Deep Representations 28 Jun 2018](https://arxiv.org/pdf/1706.01350.pdf)
<br/>
<br/>
[Averaging Weights Leads to Wider Optima and Better Generalization 25 Feb 2019](https://arxiv.org/pdf/1803.05407.pdf)
<br/>
<br/>
[Towards a Mathematical Understanding of Neural Network-Based Machine Learning 1 Oct 2020](https://arxiv.org/pdf/2009.10713v2.pdf)
<br/>
<br/>
<br/>
## [Computer vision](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/Computer_vision.rst)
Semantic segmentation, object localization
<br/>
<br/>
<br/>
[Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/pdf/1411.5752.pdf)
<br/>
<br/>
[On the Benefit of Adversarial Training for Monocular Depth Estimation 29 Oct 2019](https://arxiv.org/pdf/1910.13340.pdf)
<br/>
<br/>
<br/>
## [Transfer learning](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/Transfer_learning.rst)
Knowledge transfer, distilation, domain adaptation
<br/>
<br/>
<br/>
[Data augmentation using generative adversarial networks (CycleGAN) to improve generalizability in CT segmentation tasks](https://www.nature.com/articles/s41598-019-52737-x)
<br/>
<br/>
## [Synthetic data](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/Synthetic_data.rst)
Generating and using synthetic data to improve performance on real data
<br/>
<br/>
## [GANs](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/GANs.rst)
Generative adversarial network evolution, tips on training
<br/>
<br/>
[Generative Adversarial Networks 10 Jun 2014](https://arxiv.org/abs/1406.2661)
<br/>
<br/>
[Conditional Generative Adversarial Nets 6 Nov 2014](https://arxiv.org/abs/1411.1784)
<br/>
<br/>
[Improved Techniques for Training GANs 10 Jun 2016](https://arxiv.org/pdf/1606.03498.pdf)
<br/>
Feature matching - match intermediate discriminator layer statistics (mean) for real and generated examples<br/>
Minibatch discrimination - compute closeness of examples in a batch and use it as additional info for discriminator<br/>
Historical averaging - regularize weights to be not-very-far (L2) from previous weights of a network<br/>
One-sided label smoothing - smooth (change) labels of positive examples to alpha (0.9)<br/>
<br/>
[Deeplearning.ai specialization: Generative Adversarial Networks](https://www.coursera.org/specializations/generative-adversarial-networks-gans)
<br/>
<br/>
[Conditional Generative Adversarial Nets 6 Nov 2014](https://arxiv.org/abs/1411.1784)
<br/>
<br/>
[Deep Convolutional Generative Adversarial Networks 19 Nov 2015](https://arxiv.org/abs/1511.06434)
<br/>
<br/>
[Wasserstein GAN 6 Dec 2017](https://arxiv.org/pdf/1701.07875.pdf)
<br/>
<br/>
[Improved Training of Wasserstein GANs 31 Mar 2017](https://arxiv.org/abs/1704.00028)
<br/>
<br/>
[From GAN to WGAN 20 Aug 2017](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)
<br/>
<br/>
[Conditional Image Synthesis with Auxiliary Classifier GANs 20 Jul 2017](https://arxiv.org/pdf/1610.09585.pdf)
<br/>
<br/>
[GANs for Biological Image Synthesis 12 Sep 2017](https://arxiv.org/pdf/1708.04692.pdf)
<br/>
<br/>
[PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION 26 Feb 2018](https://arxiv.org/pdf/1710.10196.pdf)
<br/>
<br/>
[Image-to-Image Translation with Conditional Adversarial Networks 26 Nov 2018](https://arxiv.org/pdf/1611.07004.pdf)
<br/>
<br/>
[GAN DISSECTION: VISUALIZING AND UNDERSTANDING GENERATIVE ADVERSARIAL NETWORKS 8 Dec 2018](https://arxiv.org/pdf/1811.10597.pdf)
<br/>
<br/>
[SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS 16 Feb 2018](https://arxiv.org/pdf/1802.05957.pdf)
<br/>
<br/>
[Image-to-Image Translation with Conditional Adversarial Networks 26 Nov 2018](https://arxiv.org/pdf/1611.07004.pdf)
<br/>
<br/>
[DATA AUGMENTATION GENERATIVE ADVERSARIAL NETWORKS 21 Mar 2018](https://arxiv.org/pdf/1711.04340.pdf)
<br/>
<br/>
[High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs 20 Aug 2018](https://arxiv.org/pdf/1711.11585.pdf)
<br/><br/>
[Self-Attention Generative Adversarial Networks 14 Jun 2019](https://arxiv.org/pdf/1805.08318.pdf)
Self-attention modules + Spectral normalization + separate learning rates.
<br/>Self attention:
<br/>input: x
<br/> f(x) = Wf*x
<br/> g(x) = Wg*x
<br/> beta = row-wise softmax aplied to f(x).T*g(x)
<br/> h(x) = Wh*x
<br/> u(x) = Wu*x
<br/> attention map o = u(beta*h(x))
<br/> output: a*o + x
<br/><br/>
[LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS 25 Feb 2019](https://arxiv.org/pdf/1809.11096.pdf)
<br/><br/>
[A Large-Scale Study on Regularization and Normalization in GANs 14 May 2019](https://arxiv.org/pdf/1807.04720.pdf)
<br/><br/>
[A Style-Based Generator Architecture for Generative Adversarial Networks 29 Mar 2019](https://arxiv.org/pdf/1812.04948.pdf)
<br/><br/>
[Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints 16 Aug 2019](https://arxiv.org/pdf/1811.08180.pdf)
<br/>
<br/>
[LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS 25 Feb 2019](https://arxiv.org/pdf/1809.11096.pdf)
<br/>
<br/>
[Analyzing and Improving the Image Quality of StyleGAN 23 Mar 2020](https://arxiv.org/pdf/1912.04958.pdf) 
<br/>
<br/>
[Training Generative Adversarial Networks with Limited Data 7 Oct 2020](https://arxiv.org/abs/2006.06676)
<br/>
<br/>
[Interpreting the Latent Space of GANs for Semantic Face Editing 25 Jul 2019](https://arxiv.org/abs/1907.10786)
<br/>
<br/>
[Fast Fréchet Inception Distance 29 Sep 2020](https://arxiv.org/pdf/2009.14075.pdf)
<br/>
<br/>
[Pros and Cons of GAN Evaluation Measures 9 Feb 2018](https://arxiv.org/abs/1802.03446)
<br/>
<br/>
[Large Scale GAN Training for High Fidelity Natural Image Synthesis 28 Sep 2018](https://arxiv.org/abs/1809.11096)
<br/>
<br/>
[HYPE: A Benchmark for Human eYe Perceptual Evaluation of Generative Models 1 Apr 2019](https://arxiv.org/abs/1904.01121)
<br/>
<br/>
[Improved Precision and Recall Metric for Assessing Generative Models 15 Apr 2019](https://arxiv.org/abs/1904.06991)
<br/>
<br/>
[GANILLA: Generative Adversarial Networks for Image to Illustration Translation 13 Feb 2020](https://arxiv.org/abs/2002.05638)
<br/>
<br/>

<br/>
(https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732)
(https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/)
(https://github.com/sahilkhose/Generative-Adversarial-Networks-GANs-Specialization)
(https://arxiv.org/pdf/1807.10088.pdf)
(https://arxiv.org/pdf/1802.10560.pdf)
(http://wscg.zcu.cz/wscg2016/full/F71-full.pdf)
(https://arxiv.org/pdf/2008.02796.pdf)
(https://arxiv.org/pdf/1812.08352.pdf)
(https://arxiv.org/pdf/1905.01164.pdf)
(https://arxiv.org/pdf/1803.01229.pdf)
(https://arxiv.org/pdf/1905.08233.pdf)
(https://arxiv.org/pdf/1903.07291.pdf)
(https://deeppop.github.io/resources/robinson2017-deeppop.pdf)
(https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/)

<br/>
[Image Augmentations for GAN Training 4 Jun 2020](https://arxiv.org/abs/2006.02595#google)
<br/>
<br/>
[Navigating the GAN Parameter Space for Semantic Image Editing 1 Dec 2020](https://arxiv.org/pdf/2011.13786.pdf)
<br/>
<br/>
<br/>
<br/>
<br/>
## [Other](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/Other.rst)
<br/>
<br/>
<br/>
[THE COST OF TRAINING NLP MODELS: A CONCISE OVERVIEW 19 Apr 2020](https://arxiv.org/pdf/2004.08900.pdf)
<br/>
<br/>
[On the Measure of Intelligence 25 Nov 2019](https://arxiv.org/pdf/1911.01547.pdf)
<br/>
<br/>
[Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org/)
<br/>
<br/>
[Math as code](https://github.com/Jam3/math-as-code/)
<br/>
<br/>
[Math Snippets](https://github.com/terkelg/math)
<br/>
<br/>
## [Blogs](https://github.com/lrunaways/awesome-deep-learning-mustreads/blob/master/topics/blogs)
Great articles, blogs and awesome lists
<br/>
<br/>
<br/>
[PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
<br/>
<br/>
[Distill blog](https://distill.pub/)
<br/>
<br/>
[OpenAI blog](https://openai.com/blog/)
<br/>
<br/>
[Google AI blog](https://ai.googleblog.com/)
<br/>
<br/>
[Neurohive](https://neurohive.io/)
<br/>
<br/>
[Towards data science](https://towardsdatascience.com)
<br/>
<br/>

<br/><br/><br/>
--- add ---
Autoaugment: Learning augmentation strategies from data
Adversarial autoaugment
Towards principled methods for training generative adversarial networks.
Self-supervised gans via auxiliary rotation loss
Gans trained by a two time-scale update rule converge to a local nash equilibrium
On the "steerability" of generative adversarial networks
Are gans created equal? A large-scale study
Spectral normalization for generative adversarial networks
Self-attention generative adversarial networks.
Unsupervised data augmentation for consistency training
Amortised map inference for image super-resolution
A simple framework for contrastive learning of visual representations.
A holistic approach to semi-supervised learning
Foreground-aware Semantic Representations for Image Harmonization
Foreground-aware Semantic Representations for Image Harmonization
https://habr.com/ru/post/527860/
Freeze the discriminator: a simple baseline for fine-tuning GANs
Towards principled methods for training generative adversarial networks.
StarGAN v2: Diverse image synthesis for multiple domains.
Progressive growing of GANs for improved quality, stability, and variation.
Navigating the GAN Parameter Space for Semantic Image Editing
https://arxiv.org/pdf/1909.13719.pdf
[Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/pdf/2006.09661v1.pdf)
[Stylized Neural Painting](https://arxiv.org/pdf/2011.08114v1.pdf)
[A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)

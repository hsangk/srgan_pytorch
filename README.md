
https://github.com/eriklindernoren/PyTorch-GAN.git 참고 lr : 512x512 -> hr : 2048x2048 복원  


## Train 
![image](https://user-images.githubusercontent.com/102145595/197937049-7c797bd9-d5e6-488d-b62d-94f9cc0314ee.png)



## Test
![image](https://user-images.githubusercontent.com/102145595/197937102-9ede5693-8f92-4cf2-bd34-4b0a83c680bc.png)

train data가 많이 필요한 GAN model 특성상 1000개의 데이터로 train시켜 성능이 잘 나오지 않은 것 같다.


## Super-Resolution GAN
_Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_

### Authors
Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi

### Abstract
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

[[Paper]](https://arxiv.org/abs/1609.04802)

<p align="center">
    <img src="http://eriklindernoren.se/images/superresgan.png" width="640"\>
</p>


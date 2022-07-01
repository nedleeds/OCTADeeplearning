# OCTADeeplearning

> This repository mainly about deep learning parts which is consists of 4 parts([main](#M), [data](#D), [train](#R), [test](#E)).</br>
> The data pre-processing part is able to check in this 
> [Github Page](https://github.com/nedleeds/OCTAPreprocessing).</br>

1. **Environment Setting**</br>

Basically, deeplearning environment needs to consider lots of things.
Like, verision of cuda, nvidia driver and the Deep learning framework.
So, it is highly recommended to use docker.
I also made my experiment environment by utilizing the docker.
The fundamental environment for this experiment is like below.
> - Ubuntu (Linux OS for using Nvidia docker)
> - pytorch v1.11.0
> - cuda 11.3
> - cudnn 8  

It's little bit tricky unless download these seperately.</br>
But, you don't need to be worry about this,
Check the [dockerfile](https://github.com/nedleeds/OCTADeeplearning/blob/main/Dockerfile) 
above and use it.
```dockerfile
Dockerfile
 ```
You can also download the docker image through the 
[dockerhub](https://hub.docker.com/r/paulcurk/octa3d/tags).</br>
The basic usage of this file is consists of 2 steps (build & run).
Each command are operated on the shell prompt.
- Build example
> ```python
>  docker build . -t octa3d
> ```
- Run example
> ```python
>  docker run -d -it \
>  -v /data:/root/Share/OCTA3d/data \ 
>  -v /home/Project/OCTA3d:/root/Share/OCTA3d \
>  --name "octa3d" \
>  --rm --gpus all octa3d:latest
> ```
</br>

2. **[Main](https://github.com/nedleeds/OCTADeeplearning/blob/main/main.py)** <a id="M"></a></br>

The main function depicts overall process.
Using Data_Handler in [data.py](https://github.com/nedleeds/OCTADeeplearning/blob/main/data.py),
the input data for the learning has been set up.
All the arguements from the argparser has been described in the main.py script.

3. **[Data](https://github.com/nedleeds/OCTADeeplearning/blob/main/data.py)** <a id="D"></a></br>

The data.py is for handling dataset. 
From the pre-processing(split patch images for 2D, clipping for 3D normalizing)
to customize Pytorch's Dataset. 
I was needed to do this task for each different dimension respectively.
The concrete detail is described on the script through the comments.

4. **[Train](https://github.com/nedleeds/OCTADeeplearning/blob/main/train.py)** <a id="R"></a></br>

Classification, Autoencoder pre-training (by customizing [Clinicadl](https://clinicadl.readthedocs.io/en/latest/Train/Details/) method)
Basically we utilize the pre-invented CNN models as they've been proved it's performence.
The point is, utilizing with our pre-processing method, we could get the increased inference scores.
The models that we have used for are depicted below table.</br>

Dimension | VGGNet | ResNet | Inception V3 Net | Efficient Net | Vision Transformer |
:----:|:----:|:----:|:----:|:----:|:----:|
2D | [16, 19](https://github.com/nedleeds/OCTADeeplearning/blob/bf05a4042c9842c3311cc87049930819c78d29e8/model.py#L80) | [50, 152](https://github.com/nedleeds/OCTADeeplearning/blob/bf05a4042c9842c3311cc87049930819c78d29e8/model.py#L46) | [O](https://github.com/nedleeds/OCTADeeplearning/blob/bf05a4042c9842c3311cc87049930819c78d29e8/model.py#L149) | [O](https://github.com/nedleeds/OCTADeeplearning/blob/bf05a4042c9842c3311cc87049930819c78d29e8/model.py#L183) | [O](https://github.com/nedleeds/OCTADeeplearning/blob/bf05a4042c9842c3311cc87049930819c78d29e8/model.py#L217) |
3D | [16](https://github.com/nedleeds/OCTADeeplearning/blob/main/model.py#L287) | [18, 50](https://github.com/nedleeds/OCTADeeplearning/blob/main/utils/resnet.py#L217) | [O](https://github.com/nedleeds/OCTADeeplearning/blob/0c1fc1d55504d139ff6c86c8c8dc10b7ac538b95/utils/INCEPT_V3_3D.py#L15) | [O](https://github.com/nedleeds/OCTADeeplearning/blob/main/train.py#L17) | X |

There are several libraries to use these models and they actually automatically downloaded by provided Dockerfile.
For the paper, we utilize the VGG19, ResNet-50,152, Inception V3 for 2D and ResNet 18, 50, Inception V3 for 3D.
Because these models have been proved to be useful for the retina disease classifcation by previous researches.
After taking [binary-classification](https://github.com/nedleeds/OCTADeeplearning/blob/main/train.py#L299),
it was able to verify that retaining volumetric information has a higher performance.
<img width="980" alt="image" src="https://user-images.githubusercontent.com/48194000/176853430-bab03659-68ea-493c-b5d3-fbe87a4dec6f.png">


To leverage the transfer learning, adapt the autoencoder structure for pre-training and use the encoder parts 
for the classification with the fully connected layer. As pre-invented transfer learning method is actually using
the model parameters which come from the natural image. 
To match the given medical data and overcome the aforementioned limitation, this architecture should be applied.

<p align="center">
<img width="702" alt="image" src="https://user-images.githubusercontent.com/48194000/176852937-01c95682-b706-4d45-94e0-fcbc57fc52a6.png">
</p>


Currently, The multi-classification module has been tested and these will be combined with binary-classification
with only the classification module. As their difference is just the way of scoring.
Sooner these are integrated.
 

5. **[Test](https://github.com/nedleeds/OCTADeeplearning/blob/main/test.py)** <a id="E"></a></br>

For the testset which had been splitted about 30% from the total data was used for the extracted best models.
To explain the classification process of the extracted model, we visualize them 
by the Grad-CAM (by customizing [M3d-cam](https://github.com/MECLabTUDA/M3d-Cam))
As 3D volumetric data is used, the Grad-CAM has been customized to expand the dimension from 2D to 3D.
Overall process is like below.  
<img width="940" alt="image" src="https://user-images.githubusercontent.com/48194000/176857209-ab9e1fcd-fa97-4de6-897d-1de8357ec912.png">

After this process, improved retina lesion detection has been watched.
<img width="940" alt="image" src="https://user-images.githubusercontent.com/48194000/176857209-ab9e1fcd-fa97-4de6-897d-1de8357ec912.png">


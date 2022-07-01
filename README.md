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
The basic usage of this file is consists of 2 steps.
Each command are operated on the shell prompt.
- Build
> ```python
>  docker build . -t octa3d
> ```
- Run
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


4. **[Train](https://github.com/nedleeds/OCTADeeplearning/blob/main/train.py)** <a id="R"></a></br>

Classification, Autoencoder pre-training (by customizing [Clinicadl](https://clinicadl.readthedocs.io/en/latest/Train/Details/) method)

5. **[Test](https://github.com/nedleeds/OCTADeeplearning/blob/main/test.py)** <a id="E"></a></br>

Test the testset and base-on saved model, </br>
we visualize it by the Grad-CAM (by customizing [M3d-cam](https://github.com/MECLabTUDA/M3d-Cam)) 

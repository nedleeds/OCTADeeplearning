# OCTADeeplearning

> This repository mainly about deep learning parts which is consists of 4 parts(main, data,train,test).</br>
> The data pre-processing part is able to check in this [Github Page](https://github.com/nedleeds/OCTAPreprocessing).</br>
1. **Environment Setting**
> Basically, deeplearning environment needs to consider lots of things</br>
> Like, verision of cuda, nvidia driver and the Deep learning framework</br>
> So, it is highly recommended to use docker</br>
> I also made my experiment environment by utilizing the docker</br>
> The fundamental environment for this experiment is like below
>> - Ubuntu (Linux OS for using Nvidia docker)
>> - pytorch v1.11.0
>> - cuda 11.3
>> - cudnn 8</br>

> It's little bit tricky unless download these seperately</br>
> But, you don't need to be worry about this</br>
> Check the dockerfile above and use it
> ```dockerfile
> Dockerfile
> ```
> You can also download the docker image through the [dockerhub](https://hub.docker.com/r/paulcurk/octa3d/tags)</br>

> The basic usage of this file is consists of 2 steps
>> - Build
>> - Run</br>

2. **Main** 
> Customize the datahandler and dataset

3. **Data** 
> Customize the datahandler and dataset

4. **Train** 
> Classification, Autoencoder pre-training (by customizing [Clinicadl](https://clinicadl.readthedocs.io/en/latest/Train/Details/) method)

5. **Test** 
> Test the testset and base-on saved model, </br>
> we visualize it by the Grad-CAM (by customizing [M3d-cam](https://github.com/MECLabTUDA/M3d-Cam))
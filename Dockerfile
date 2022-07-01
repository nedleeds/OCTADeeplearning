FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
WORKDIR /root/Share/OCTA3d
ENV DEBIAN_FRONTEND noninteractive


RUN apt-get update &&\
    apt-get install -y --no-install-recommends apt-utils \
    sudo -y \
    -y zsh \
    libgl1-mesa-glx -y \
    libncurses5-dev -y \
    libncursesw5-dev -y \
    git -y \
    make \
    build-essential \
    wget \
    libx11-dev -y \
    vim -y \
    libglib2.0-0 -y \
    python3-tk -y &&\
    sudo apt-get install libboost-python-dev build-essential -y \
    libpcre3 libpcre3-dev

RUN apt-get install python3 python-dev python3-dev \
    build-essential libssl-dev libffi-dev -y\
    libxml2-dev libxslt1-dev zlib1g-dev -y\
    tmux
    
RUN pip install --upgrade pip &&\
    pip install \
    scipy \
    numba \
    numpy \
    ipython \
    ipykernel \
    pandas \
    Image \
    matplotlib \
    sklearn \
    gpustat \
    keras \
    tensorboard \
    SimpleITK \
    times \
    nibabel \
    scikit-image \ 
    medpy \ 
    openpyxl \
    torchsummary \
    opencv-python \
    seaborn \
    medcam \
    pytorch-lightning \
    ray \
    flaml \
    git+https://github.com/shijianjian/EfficientNet-PyTorch-3D \
    vit-pytorch \
    efficientnet_pytorch




RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t cloud \
    -p git \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-history-substring-search \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p 'history-substring-search' \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'

RUN sh -s /bin/zsh
RUN PATH="$PATH:~/bin/zsh:/usr/bin/zsh:/bin/zsh/:/zsh:/Applications/Visual Studio Code.app/Contents/Resources/app/bin"
RUN echo "ZSH_THEME_CLOUD_PREFIX=\"???  ???%{\$fg[green]%} :%\"" \
    "PROMPT='%{\$fg_bold[green]%}\$ZSH_THEME_CLOUD_PREFIX %{\$fg_bold[green]%} %{\$fg[magenta]%}%c %{\$fg_bold[cyan]%}\$(git_prompt_info)%{\$fg_bold[blue]%}%{\$reset_color%}'" \
    "ZSH_THEME_GIT_PROMPT_PREFIX=\"%{\$fg[blue]%}[%{\$fg[green]%}\"" \
    "ZSH_THEME_GIT_PROMPT_SUFFIX=\"%{$reset_color%}\"" \
    "ZSH_THEME_GIT_PROMPT_DIRTY=\"%{\$fg[blue]%}] %{\$fg[green]%}???%{\$reset_color%}\"" \
    "ZSH_THEME_GIT_PROMPT_CLEAN=\"%{\$fg[blue]%}] ???\"" \
    "PROMPT+=$'\n%{\$fg[white]%}?? '" \
    "alias python=\"python3\"" >> ~/.zshrc


CMD ["/bin/zsh"]

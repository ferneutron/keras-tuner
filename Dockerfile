FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y sudo curl gnupg wget && \
    useradd -ms /bin/bash app && \
    echo "app ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER app

# Install Python packages together
RUN sudo pip install --upgrade pip
RUN sudo pip install keras==2.15.0 \
                     keras-tuner==1.4.7 \
                     tensorflow==2.15.0 \
                     tensorboard==2.15.2
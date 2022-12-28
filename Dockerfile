FROM pytorchlightning/pytorch_lightning:base-cuda-py3.6-torch1.8

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /root

COPY ./ /root/imudb

WORKDIR /root/imudb
RUN pip install catkin_pkg
RUN pip install git+https://github.com/eric-wieser/ros_numpy.git
RUN pip install -r requirements.txt
RUN pip install --extra-index-url https://rospypi.github.io/simple/ rosbag

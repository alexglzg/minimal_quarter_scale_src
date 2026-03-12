FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    python3-tk \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo (if not included in base image)
RUN apt-get update && apt-get install -y \
    gazebo11 \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-pcl-ros \
    ros-noetic-ddynamic-reconfigure \
    ros-noetic-robot-localization \
    ros-noetic-move-base \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from your pip_packages.txt
RUN pip3 install --no-cache-dir \
    casadi \
    rockit-meco \
    numpy \
    matplotlib \
    transforms3d \
    packaging

RUN apt-get update && apt-get install -y \
    ros-noetic-serial \
    ros-noetic-velodyne-description \
    ros-noetic-velodyne-simulator \
    libnlopt-dev

# Set up workspace
RUN mkdir -p /ros1_ws/src
WORKDIR /ros1_ws

# Source ROS setup in bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /ros1_ws/devel/setup.bash" >> ~/.bashrc

CMD ["bash"]

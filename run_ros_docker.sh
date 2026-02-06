#!/bin/bash

xhost +local:docker

WORKSPACE=$(pwd)

# Build the docker image
docker build -t ros1_noetic .

# # Check if container already exists
# if [ "$(docker ps -aq -f name=noetic)" ]; then
    # echo "Container 'noetic' already exists!"
    # echo "Use: docker start noetic && docker exec -it noetic bash"
    # exit 1
# fi

# Run the Docker container - NOTE: This must be ONE command, no line breaks!
docker run -itd \
     --name noetic \
     --net=host \
     --privileged \
     -e DISPLAY=$DISPLAY \
     -e QT_X11_NO_MITSHM=1 \
     --device=/dev/dri:/dev/dri \
     --gpus all \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     -v $WORKSPACE:/ros1_ws/src:rw \
     ros1_noetic

echo "Container created. Enter with: docker exec -it noetic bash"

xhost -local:docker

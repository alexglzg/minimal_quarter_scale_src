#!/bin/bash
xhost +local:docker
docker start noetic
docker exec -it noetic bash

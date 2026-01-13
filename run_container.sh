docker run -it \
    --rm \
    --name quarterscale \
    --net=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MISTHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$(pwd)":/ros1_ws/src \
    quarterscale bash
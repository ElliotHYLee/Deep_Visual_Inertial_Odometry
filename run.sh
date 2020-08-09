# docker rm pytorch_cnt
docker build -t dvio_img .
docker run -it --rm --gpus all \
-v $(pwd)/src:/workspace/src \
-v /media/el/SSD/KITTI:/workspace/KITTI/ \
-e JUPYTER_TOKEN=asdf \
-p 8888:8888 \
--name=dvio_cnt \
dvio_img 

# -v /tmp/.X11-unix/:/tmp/.X11-unix/:rw \
# -v $HOME/.Xauthority:/root/.Xauthority:rw \
# -e DISPLAY=${DISPLAY} \
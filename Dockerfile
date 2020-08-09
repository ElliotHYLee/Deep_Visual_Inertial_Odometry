FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev
WORKDIR /workspace/src
RUN python -m pip install jupyter
RUN pip install numpy scipy pandas matplotlib scikit-learn pathlib opencv-python opencv-contrib-python 


RUN jupyter notebook --generate-config
RUN echo "c=get_config()" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.InlineBackend.rc = { }" >> /root/.jupyter/jupyter_notebook_config.py

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
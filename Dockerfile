FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && sh ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

COPY pyproject.toml pyproject.toml
COPY diffusion diffusion
COPY configs configs
COPY sana sana
COPY app app
COPY tools tools

COPY environment_setup.sh environment_setup.sh
RUN ./environment_setup.sh

ENV HF_HOME=/huggingface

#CMD ["python", "-u", "-W", "ignore", "app/app_sana.py", "--share", "--config=configs/sana_config/512ms/Sana_1600M_img512.yaml", "--model_path=hf://Efficient-Large-Model/Sana_1600M_512px/checkpoints/Sana_1600M_512px_MultiLing.pth"]


£CMD ["python", "-u", "-W", "ignore", "app/app_sana.py", "--share", "--config=configs/sana_config/512ms/Sana_600M_img512.yaml", "--model_path=hf://Efficient-Large-Model/Sana_600M_512px/checkpoints/Sana_600M_512px_MultiLing.pth"]
CMD ["python", "-u", "-W", "ignore", "app/app_sana.py", "--share", "--config=configs/
sana_sprint_config/1024ms/SanaSprint_600M_1024px_allqknorm_bf16_scm_ladd.yaml", "--model_path=hf://Efficient-Large-Model/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth"]

#CMD ["python", "-u", "-W", "ignore", "app/app_sana.py", "--share", "--config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml", "--model_path=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"]
#CMD ["python", "-u", "-W", "ignore", "app/app_sana.py", "--share", "--config=configs/sana1-5_config/1024ms/
Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml", "--model_path=hf://Efficient-Large-Model/SANA1.5_1.6B_1024px/blob/main/checkpoints/SANA1.5_1.6B_1024px.pth"]

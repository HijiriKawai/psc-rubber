FROM sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1

USER root

RUN apt-get update \
  && apt-get -y --no-install-recommends install graphviz=2.42.2-3build2 \
  && apt-get clean \  
  && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash researcher \
  && chown -R researcher /home/researcher \
  && pip install --no-cache-dir  pandas==1.5.2 seaborn==0.12.1 scikit-learn==1.2.0 keras_self_attention==0.51.0 flake8==6.0.0 

WORKDIR /home/researcher

COPY . .

USER researcher

CMD [ "/bin/bash" ]

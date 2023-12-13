FROM sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1

USER root

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN wget --progress=dot:giga -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
  && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

RUN apt-get update \
  && apt-get -y --no-install-recommends install graphviz=2.42.2-3build2 \
  && apt-get clean \  
  && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash researcher \
  && chown -R researcher /home/researcher \
  && pip install --no-cache-dir  pandas==1.5.2 seaborn==0.12.1 scikit-learn==1.2.0 keras_self_attention==0.51.0 flake8==6.0.0 pydot==1.4.2 graphviz==0.20.1 xlsx2csv==0.8.1 transformers==4.25.1 fastprogress==0.2.3

WORKDIR /home/researcher

COPY . .

USER researcher

CMD [ "/bin/bash" ]

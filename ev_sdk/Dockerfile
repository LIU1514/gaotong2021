FROM uhub.service.ucloud.cn/eagle_nest/ubuntu16.04-cuda10.0-cudnn7.4-opencv4.1-snpe1.47-tensorflow1.13

RUN rm -rf /usr/local/ev_sdk && mkdir -p /usr/local/ev_sdk
COPY ./ /usr/local/ev_sdk
RUN pip install sklearn
RUN pip install scikit-image
RUN python3.6 -m pip install sklearn
RUN \
    cd /usr/local/ev_sdk && mkdir -p build && rm -rf build/* \
    && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4 install && rm -rf ../build/*

ENV AUTO_TEST_USE_JI_PYTHON_API=1
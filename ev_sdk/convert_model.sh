#!/bin/bash
# 在这里编写OpenVINO模型转换
# 系统会将所选择的原始模型放在目录/usr/local/ev_sdk/model下，转换后的OpenVINO模型请放在model/openvino目录下，
# 建议所有路径都使用绝对路径
# 获取当前脚本的绝对路径
pip install sklearn
pip install scikit-learn
pwd
python3.6 -m pip install scikit-learn
echo "path..." 
ls /usr/local/ev_sdk
ls /usr/local/ev_sdk/test
ls /usr/local/ev_sdk/data
ls /usr/local/ev_sdk/model

echo "path..." 
ls /usr/local/ev_sdk
ls /usr/local/ev_sdk/model
python /usr/local/ev_sdk/model/test.py
echo "Start training..." \

source /opt/snpe/snpe_venv/bin/activate
pip install sklearn
pip install scikit-learn
python3.6 -m pip install scikit-learn
snpe-tensorflow-to-dlc \
--input_network /usr/local/ev_sdk/model/model200.pb \
--input_dim input "1,100,100,3" \
--out_node "y_conv" \
--output_path /usr/local/ev_sdk/model/model200.dlc
#snpe-dlc-info -i /usr/local/ev_sdk/model/model200.dlc
snpe-dlc-quantize \
--input_dlc /usr/local/ev_sdk/model/model200.dlc \
--input_list /project/ev_sdk/raw_list.txt \
--output_dlc /usr/local/ev_sdk/model/model200Q.dlc \
--input_dlc --enable_htp
#snpe-dlc-info -i /usr/local/ev_sdk/model/model200Q.dlc
rm /usr/local/ev_sdk/model/model200.dlc
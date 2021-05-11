project_root_dir=/project/train/src_repo
###
 # @Author: your name
 # @Date: 2021-05-07 09:35:33
 # @LastEditTime: 2021-05-11 10:48:26
 # @LastEditors: your name
 # @Description: In User Settings Edit
 # @FilePath: \raw\train\train.sh
### 
dataset_dir=/home/data
log_file=/project/train/log/log.txt
pip install -i https://mirrors.aliyun.com/pypi/simple -r /project/train/src_repo/requirements.txt \
&& pip list \
&& echo "Start training..." \
&& cd ${project_root_dir} && python -u traintf.py | tee -a ${log_file} \
&& echo "Done" 

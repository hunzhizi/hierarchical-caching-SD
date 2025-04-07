class Config():
    _autodl_dir = "/root/autodl-tmp/model"
    _heng_yuan_yun = "/hy-tmp"
    _3090ti = "/mnt/data/zhouShaoRepo/model"
    PREDICTION_NUM = 1
    MODEL_NUM = 3
    MAX_LEN = 512
    BUFFER_SIZE = MAX_LEN + 50
    END_FLAG = 10
    MODEL_DIR = _heng_yuan_yun
    IS_BRANCH_PREDICTION = True
    PORJECT_ROOT_DIR="/root/hierarchical-sd"
    PROFILE_MODEL_INFERENCE_DIR=PORJECT_ROOT_DIR +"/benchmark/profile_inference"

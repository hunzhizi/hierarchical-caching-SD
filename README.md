CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3 TestModel.py --eval_mode para_sd  -n 1  -e H_PSD_codellama_7_70b --draft_model Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct --target_model Qwen2.5-7B-Instruct --max_tokens 512 --temp 0

* 服务器运行需要指定python搜索环境 
* # 假设项目根目录是 /tmp/pycharm_project_329
export PYTHONPATH=$PYTHONPATH:/root/hierarchical-sd
* 指定 transformers 版本为 pip install transformers==4.45.2
* pip3 install transformers==4.45.2 tqdm ipdb accelerate numpy shortuuid fschat fastchat

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3 --multi_gpu --num_machines 1 --gpu_ids 0,1,2  TestModel.py --eval_mode para_sd  -n 1  -e H_PSD_codellama_7_70b --draft_model Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct --target_model Qwen2.5-7B-Instruct --max_tokens 512 --temp 0
* 测试单个模型
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 1  --num_machines 1  TestModel.py --eval_mode single_model --model_name Qwen2.5-7B-Instruct --max_tokens 512 --temp 0
* 使用Qwen2.5 parallel decoding 进行推理
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 TestModelCpuCentric.py --eval_mode para_sd --draft_model Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct --target_model Qwen2.5-7B-Instruct --max_tokens 512 

* 使用 llama3 parallel decoding 进行推理
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 TestModelCpuCentric.py --eval_mode para_sd --draft_model Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct --target_model Llama-3.1-8B-Instruct --max_tokens 512 
* 测试单个模型 todo 未测试
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 TestModelCpuCentric.py --eval_mode single_model --model_name Llama-3.1-8B-Instruct --max_tokens 512 


CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3 TestModel.py --eval_mode para_sd  -n 1  -e H_PSD_codellama_7_70b --draft_model Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct --target_model Qwen2.5-7B-Instruct --max_tokens 512 --temp 0

* 服务器运行需要指定python搜索环境 
* # 假设项目根目录是 /tmp/pycharm_project_329
export PYTHONPATH=$PYTHONPATH:/tmp/pycharm_project_58

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3 --multi_gpu --num_machines 1 --gpu_ids 0,1,2  TestModel.py --eval_mode para_sd  -n 1  -e H_PSD_codellama_7_70b --draft_model Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct --target_model Qwen2.5-7B-Instruct --max_tokens 512 --temp 0

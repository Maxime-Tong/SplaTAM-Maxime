#!/bin/bash
# 配置参数
MIN_GPU_MEMORY=20000  # 最小需要的GPU显存(MB)，根据你的需求调整
datasets=(0 1 2 3 4 5 6 7)
args_combinations=(
    "8 10 uniform random_texture_flip_8"
)

LOG_DIR="logs/mapping"
mkdir -p "$LOG_DIR"

config_path="configs/replica/replica_eval_test.py" # PATH TO YOUR MODELS

# 获取符合内存要求的GPU
select_gpu() {
    # 获取GPU信息并筛选满足内存要求的
    gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    
    # 筛选内存大于阈值且按内存排序
    available_gpus=$(echo "$gpu_info" | awk -F',' -v min_mem="$MIN_GPU_MEMORY" \
        '$2 > min_mem {print $1, $2}' | sort -k2 -nr)
    
    if [ -z "$available_gpus" ]; then
        echo "No GPU available with at least ${MIN_GPU_MEMORY}MB free memory" >&2
        exit 1
    fi
    
    # 选择剩余内存最多的GPU
    best_gpu=$(echo "$available_gpus" | head -n1 | awk '{print $1}')
    echo "$best_gpu"
}

# 执行任务函数
run_task() {
    local dataset_name=$1
    local combo=$2
    local gpu_id=$3
    local port=$4
    
    local t_scale=$(echo $combo | awk '{print $1}')
    local m_scale=$(echo $combo | awk '{print $2}')
    local t_fn=$(echo $combo | awk '{print $3}')
    local m_fn=$(echo $combo | awk '{print $4}')
    

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local group_name="${t_fn}_${t_scale}_${m_fn}_${m_scale}"
    local log_prefix="${LOG_DIR}/${group_name}_${dataset_name}"

    echo "Running scene ${dataset_name} (${group_name}) on GPU ${gpu_id}" | tee -a "${log_prefix}.log"
    {
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export SCENE_NUM=${dataset_name}
        export GROUP_NAME=${group_name}

        python3 -u scripts/splatam_test.py configs/replica/replica_eval_test.py --tracking_fn $t_fn --tracking_scale $t_scale --mapping_fn $m_fn --mapping_scale $m_scale
        
    } >> "${log_prefix}.log" 2>&1
    
    # 记录完成状态
    if [ $? -eq 0 ]; then
        echo "[$(date)] Successfully completed $dataset_name" | tee -a "${log_prefix}.log"
    else
        echo "[$(date)] Failed processing $dataset_name" | tee -a "${log_prefix}.log"
    fi
}

# 主执行循环
for dataset_name in "${datasets[@]}"; do
    for combo in "${args_combinations[@]}"
    do
        # 选择GPU
        gpu_id=$(select_gpu)
        if [ $? -ne 0 ]; then
            echo "Skipping $dataset_name due to GPU unavailability"
            continue
        fi
        
        # 在后台运行任务
        run_task "$dataset_name" "$combo" "$gpu_id" "$current_port"&
        ((current_port++))
        
        # 可选：添加延迟以避免GPU选择冲突
        sleep 60
    done
done

# 等待所有后台任务完成
wait
echo "All tasks completed. Final status:"
grep -h "Failed" "$LOG_DIR"/*.log

exit 0
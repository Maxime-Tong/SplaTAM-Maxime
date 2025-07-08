#!/bin/bash

# 获取系统中可用的GPU数量
num_gpus=$(nvidia-smi -L | wc -l)
echo "Detected $num_gpus GPUs"

# 初始化任务计数器
task_count=1

# 手动指定 tile_x 和 tile_y 的组合
args_combinations=(
    "16 8 uniform normal"
    "16 8 uniform random"
    "16 8 uniform novelty"
    "16 8 uniform novelty_random"
    # "32 32"
)

for scene in 7
do
    # (
    #     export SCENE_NUM=${scene}
    #     export GROUP_NAME="original"
    #     echo "Running scene ${SCENE_NUM} (${GROUP_NAME}) on GPU ${gpu_id}"
    #     python3 -u scripts/splatam.py configs/replica/replica_eval_test.py > logs/${GROUP_NAME}_${scene}.log 2>&1
    #     echo "Task scene ${SCENE_NUM} (${GROUP_NAME}) completed on GPU ${gpu_id}" 
    # ) &

    for combo in "${args_combinations[@]}"
    do
        t_scale=$(echo $combo | awk '{print $1}')
        m_scale=$(echo $combo | awk '{print $2}')
        t_fn=$(echo $combo | awk '{print $3}')
        m_fn=$(echo $combo | awk '{print $4}')
        
        
        gpu_id=$((task_count % num_gpus))
        export CUDA_VISIBLE_DEVICES=$gpu_id
        task_count=$((task_count + 1))
        (
            export SCENE_NUM=${scene}
            export GROUP_NAME="${t_fn}_${t_scale}_${m_fn}_${m_scale}"
            echo "Running scene ${SCENE_NUM} (${GROUP_NAME}) on GPU ${gpu_id}"
            python3 -u scripts/splatam.py configs/replica/replica_eval_test.py --tracking_fn $t_fn --tracking_scale $t_scale --mapping_fn $m_fn --mapping_scale $m_scale > logs/${GROUP_NAME}_${scene}.log 2>&1
            echo "Task scene ${SCENE_NUM} (${GROUP_NAME}) completed on GPU ${gpu_id}"
        ) &

        sleep 2
    done
done

wait
echo "All tasks completed"
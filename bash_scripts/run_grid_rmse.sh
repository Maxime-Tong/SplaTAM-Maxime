#!/bin/bash

# 获取系统中可用的GPU数量
num_gpus=$(nvidia-smi -L | wc -l)
echo "Detected $num_gpus GPUs"

# 初始化任务计数器
task_count=1

# 手动指定 tile_x 和 tile_y 的组合
tile_combinations=(
    # "8 8"
    "16 16"
    # "32 32"
)

for scene in 0 1 2 3 4 5 6
do
    for combo in "${tile_combinations[@]}"
    do
        # 分割组合为 tile_x 和 tile_y
        tile_x=$(echo $combo | awk '{print $1}')
        tile_y=$(echo $combo | awk '{print $2}')
        
        for trakcing_fn in random uniform
        do
            gpu_id=$((task_count % num_gpus))
            export CUDA_VISIBLE_DEVICES=$gpu_id
            task_count=$((task_count + 1))

            (
                export SCENE_NUM=${scene}
                export GROUP_NAME="${fn}_${tile_x}_${tile_y}_mapping_sparse"
                echo "Running scene ${SCENE_NUM} (${GROUP_NAME}) on GPU ${gpu_id}"
                python3 -u scripts/splatam.py configs/replica/replica_eval_test.py --use_sparse --tracking_fn $trakcing_fn --tile_size 16 --mapping_fn adaptive_random  > logs/${GROUP_NAME}_${scene}.log 2>&1
                echo "Task scene ${SCENE_NUM} (${GROUP_NAME}) completed on GPU ${gpu_id}"
            ) &

            sleep 2
        done
    done
done

wait
echo "All tasks completed"
# K / ML / CL 2×2×2 控制实验

注意！！以下脚本统一使用数组 + "${arr[@]}" 的写法，在 bash 与 zsh 下行为一致。

<span style="color: red">**不要用未加引号的 for x in $VAR 形式！！zsh 默认不做单词切分，会导致循环只跑一次（曾产生 80 行而非 960 行的坏清单）。**</span>

## 组合与 seed 约定

所有"样本 × 组合"统一使用同一组 seed（基数 200，重复 200–209），使同一重复号在 8 个组合间配对。

|K|ML|CL|seed|
|---|---|---|---|
|正常|T|T|200–209|
|正常|T|F|200–209|
|正常|F|T|200–209|
|正常|F|F|200–209|
|0|T|T|200–209|
|0|T|F|200–209|
|0|F|T|200–209|
|0|F|F|200–209|

```bash
cd "/hdd2/liuzy/Projects/IntensiveStudy4Lin/DSSC/src"
conda activate dssc-4090
```

## 顺序版：12 样本 × 8 组合 × 10 重复

```bash
# SAMPLES=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
# SEEDS=($(seq 200 209))                       # 10 个重复
# COMBOS=("20 1 1" "20 1 0" "20 0 1" "20 0 0" "0 1 1" "0 1 0" "0 0 1" "0 0 0")

# for sample in "${SAMPLES[@]}"; do
#     f="../data/spatialLIBD/sample_${sample}_anno.h5"
#     mlf="../example_constraints/sample_${sample}_mlFromMarks_test.txt"
#     clf="../example_constraints/sample_${sample}_clFromMarks_test.txt"
#     for c in "${COMBOS[@]}"; do
#         read K ML CL <<< "$c"
#         flags=""
#         [ "$K"  = 0 ] && flags="$flags --knn 0"
#         [ "$ML" = 0 ] && flags="$flags --no_ml"
#         [ "$CL" = 0 ] && flags="$flags --no_cl"
#         out="./out_${sample}/K${K}_ML${ML}_CL${CL}"
#         mkdir -p "$out"
#         for seed in "${SEEDS[@]}"; do
#             python -u run_DSSC.py --n_clusters -1 --data_file "$f" --save_dir "$out" \
#                 --final_labels pred.csv --final_latent_file latent.csv \
#                 --seed "$seed" --ml_file "$mlf" --cl_file "$clf" $flags
#         done
#     done
# done
```

## 双卡并行。同一任务集，两张 GPU 各跑一半

思路：把 (sample, K, ML, CL, seed) 全部展开成任务清单，对半切，两个 worker 各绑定一张卡
（CUDA_VISIBLE_DEVICES），互不干扰。无需改 src 源码（run_DSSC.py 的 --device 默认 cuda，
被 CUDA_VISIBLE_DEVICES 自动限定到可见卡）。

```bash
cd "/hdd2/liuzy/Projects/IntensiveStudy4Lin/DSSC/src"
conda activate dssc-4090

SAMPLES=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
SEEDS=($(seq 200 209))
COMBOS=("20 1 1" "20 1 0" "20 0 1" "20 0 0" "0 1 1" "0 1 0" "0 0 1" "0 0 0")

# 生成任务清单：sample K ML CL seed
: > tasks.txt
for sample in "${SAMPLES[@]}"; do
    for c in "${COMBOS[@]}"; do
        read K ML CL <<< "$c"
        for seed in "${SEEDS[@]}"; do
            echo "$sample $K $ML $CL $seed"
        done
    done
done > tasks.txt
echo "total tasks: $(wc -l < tasks.txt)" # 应为 12*8*10 = 960

# 对半切（两等份）
split -n l/2 -d tasks.txt taskpart_

# worker：读一份清单，绑定一张卡
run_worker () {
    local gpu=$1 part=$2
    while read sample K ML CL seed; do
        f="../data/spatialLIBD/sample_${sample}_anno.h5"
        mlf="../example_constraints/sample_${sample}_mlFromMarks_test.txt"
        clf="../example_constraints/sample_${sample}_clFromMarks_test.txt"
        flags=""
        [ "$K" = 0 ] && flags="$flags --knn 0"
        [ "$ML" = 0 ] && flags="$flags --no_ml"
        [ "$CL" = 0 ] && flags="$flags --no_cl"
        out="./out_${sample}/K${K}_ML${ML}_CL${CL}"
        mkdir -p "$out"
        CUDA_VISIBLE_DEVICES=$gpu python -u run_DSSC.py --n_clusters -1 \
        --data_file "$f" --save_dir "$out" \
        --final_labels pred.csv --final_latent_file latent.csv \
        --seed "$seed" --ml_file "$mlf" --cl_file "$clf" $flags
    done < "$part"
}

# 双卡并行
run_worker 0 taskpart_00 > log_gpu0.txt 2>&1 &
run_worker 1 taskpart_01 > log_gpu1.txt 2>&1 &
wait
echo "ALL DONE"
```
运行前验证

```bash
cd "/hdd2/liuzy/Projects/IntensiveStudy4Lin/DSSC/src"
SAMPLES=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
SEEDS=($(seq 200 209))
COMBOS=("20 1 1" "20 1 0" "20 0 1" "20 0 0" "0 1 1" "0 1 0" "0 0 1" "0 0 0")
# 生成任务清单：sample K ML CL seed
: > tasks.txt
for sample in "${SAMPLES[@]}"; do
    for c in "${COMBOS[@]}"; do
        read K ML CL <<< "$c"
        for seed in "${SEEDS[@]}"; do
            echo "$sample $K $ML $CL $seed"
        done
    done
done > tasks.txt

echo "total tasks: $(wc -l < tasks.txt)" # 应为 960
# wc -l tasks.txt 
awk '{print $1}' tasks.txt | sort -u | wc -l # 应为 12
head tasks.txt # 首行应为 151507 20 1 1 200
```


输出结构：`src/out_<sample>/K<K>_ML<ML>_CL<CL>/` 下每个 seed 一套 `pred.csv_<seed>`、`latent.csv_<seed>`，即 12 个样本目录，每个目录 8 个组合子目录，每个子目录 10 个重复。`out_*` 已被 `.gitignore` 忽略。
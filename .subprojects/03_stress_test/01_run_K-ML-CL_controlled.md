|K|ML|CL|seed|
|---|---|---|---|
|正常|T|T|200|
|正常|T|F|300|
|正常|F|T|400|
|正常|F|F|500|
|0|T|T|600|
|0|T|F|700|
|0|F|T|800|
|0|F|F|900|

```bash
cd "/hdd2/liuzy/Projects/IntensiveStudy4Lin/DSSC/src"
conda activate dssc-4090
```

```bash
# K = 20, ML use，CL use
# seed: 200
f="../data/sample_151507_anno.h5"
for i in {1..5}
do
    python -u run_DSSC.py --n_clusters -1 --data_file $f --save_dir out_151507 \
        --final_labels pred.csv --final_latent_file latent.csv --run $i --seed $((200+i)) \
        --ml_file "../example_constraints/sample_151507_mlFromMarks_test.txt" --cl_file "../example_constraints/sample_151507_clFromMarks_test.txt"
done
```

```bash
# K = 20, ML use，CL not use
# seed: 300
f="../data/sample_151507_anno.h5"
for i in {1..5}
do
    python -u run_DSSC.py --n_clusters -1 --data_file $f --save_dir out_151507 \
        --final_labels pred.csv --final_latent_file latent.csv --run $i --seed $((300+i)) \
        --ml_file "../example_constraints/sample_151507_mlFromMarks_test.txt" --cl_file "../example_constraints/sample_151507_clFromMarks_test.txt" \
        --no_cl
done
```

```bash
# K = 20, ML not use，CL use
# seed: 400
f="../data/sample_151507_anno.h5"
for i in {1..5}
do
    python -u run_DSSC.py --n_clusters -1 --data_file $f --save_dir out_151507 \
        --final_labels pred.csv --final_latent_file latent.csv --run $i --seed $((400+i)) \
        --ml_file "../example_constraints/sample_151507_mlFromMarks_test.txt" --cl_file "../example_constraints/sample_151507_clFromMarks_test.txt" \
        --no_ml
done
```

```bash
# K = 20, ML not use，CL not use
# seed: 500
f="../data/sample_151507_anno.h5"
for i in {1..5}
do
    python -u run_DSSC.py --n_clusters -1 --data_file $f --save_dir out_151507 \
        --final_labels pred.csv --final_latent_file latent.csv --run $i --seed $((500+i)) \
        --ml_file "../example_constraints/sample_151507_mlFromMarks_test.txt" --cl_file "../example_constraints/sample_151507_clFromMarks_test.txt" \
        --no_cl --no_ml
done
```

```bash
# K = 0, ML use，CL use
# seed: 600
f="../data/sample_151507_anno.h5"
for i in {1..5}
do
    python -u run_DSSC.py --n_clusters -1 --data_file $f --save_dir out_151507 \
        --final_labels pred.csv --final_latent_file latent.csv --run $i --seed $((600+i)) \
        --ml_file "../example_constraints/sample_151507_mlFromMarks_test.txt" --cl_file "../example_constraints/sample_151507_clFromMarks_test.txt" \
        --knn 0
done
```

```bash
# K = 0, ML use，CL not use
# seed: 700
f="../data/sample_151507_anno.h5"
for i in {1..5}
do
    python -u run_DSSC.py --n_clusters -1 --data_file $f --save_dir out_151507 \
        --final_labels pred.csv --final_latent_file latent.csv --run $i --seed $((700+i)) \
        --ml_file "../example_constraints/sample_151507_mlFromMarks_test.txt" --cl_file "../example_constraints/sample_151507_clFromMarks_test.txt" \
        --knn 0 --no_cl
done
```

```bash
# K = 0, ML not use，CL use
# seed: 800
f="../data/sample_151507_anno.h5"
for i in {1..5}
do
    python -u run_DSSC.py --n_clusters -1 --data_file $f --save_dir out_151507 \
        --final_labels pred.csv --final_latent_file latent.csv --run $i --seed $((800+i)) \
        --ml_file "../example_constraints/sample_151507_mlFromMarks_test.txt" --cl_file "../example_constraints/sample_151507_clFromMarks_test.txt" \
        --knn 0 --no_ml
done
```

```bash
f="../data/sample_151507_anno.h5"
for i in {1..5}
do
    python -u run_DSSC.py --n_clusters -1 --data_file $f --save_dir out_151507 \
        --final_labels pred.csv --final_latent_file latent.csv --run $i --seed $((900+i)) \
        --ml_file "../example_constraints/sample_151507_mlFromMarks_test.txt" --cl_file "../example_constraints/sample_151507_clFromMarks_test.txt" \
        --knn 0 --no_ml --no_cl
done
```
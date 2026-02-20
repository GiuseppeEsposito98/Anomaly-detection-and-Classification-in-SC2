target_folder=$1
sc_conf=(1 2 3 6 9 12)

for n in "${sc_conf[@]}"; do
    
    echo "**************************************"
    echo "Running configuration Split Configuration ${n}"

        python3 src/train_1.py \
            --statistic_dir runs \
            --data_root ${target_folder}/cnf"$n"/

done
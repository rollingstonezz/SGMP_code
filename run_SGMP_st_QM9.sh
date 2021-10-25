python3 main_base_st.py --save_dir ./results \
              --data_dir ./data \
              --model SGMP \
              --dataset QM9 \
              --split 811 \
              --device gpu \
              --random_seed 1 \
              --batch_size 64 \
              --epoch 500 \
              --lr 1e-3 \
              --test_per_round 5 \
              --label 0 \
              --spanning_tree True \
              --num_layers 3 
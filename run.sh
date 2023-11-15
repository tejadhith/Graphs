python main.py --graphsage_flag --dataset="Cora" --device="cuda" \
                --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
                --num_epochs 100 --num_runs 1 --lr 1e-2 --weight_decay 5e-4 --weight_decay 5e-4 --batch_size 1 \
                --output_dir "output" --file_name "SageCora14NovTry1" \
                --num_workers 32 --verbose

# python main.py --random_walk_flag --dataset "Cora" --device "cuda" \
#                --random_walk_model "DeepWalk" \
#                --embedding_dim 32 --mlp_hidden_dim 16 \
#                --walk_length 10 --num_walks 10 \
#                --num_epochs 10 --num_runs 2 --lr 0.01 --weight_decay 5e-4 --weight_decay 5e-4 --batch_size 32 \
#                --num_workers 4 --verbose \
#                --output_dir "output" --file_name "RandomWalkTrial1" 

# python main.py --random_walk_flag --dataset "Cora" --device "cuda" \
#                --random_walk_model "Node2Vec" \
#                --embedding_dim 32 --mlp_hidden_dim 16 \
#                --walk_length 10 --num_walks 10 \
#                --num_epochs 10 --num_runs 2 --lr 0.01 --weight_decay 5e-4 --weight_decay 5e-4 --batch_size 32 \
#                --num_workers 4 --verbose \
#                --output_dir "output" --file_name "RandomWalkTrial1" 

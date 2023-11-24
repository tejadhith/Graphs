# python main.py --gcn_flag --dataset="Cora" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-3 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GCNCora" \
#                 --num_workers 32 --verbose --patience 20 --model_save \

# python main.py --graphsage_flag --dataset="Cora" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-3 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GraphSageCora" \
#                 --num_workers 32 --verbose --patience 20 --model_save \

# python main.py --gat_flag --dataset="Cora" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-3 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GatCora" \
#                 --num_workers 32 --verbose --patience 20 --model_save \

# python main.py --gcn_flag --dataset="Cora" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-4 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GCNPoincareCora" \
#                 --num_workers 32 --verbose --patience 20 --model_save \
#                 --hyperbolic_flag --hyperbolic_model "poincare"

# python main.py --gcn_flag --dataset="Cora" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-4 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GCNLorentzCora" \
#                 --num_workers 32 --verbose --patience 20 --model_save \
#                 --hyperbolic_flag --hyperbolic_model "lorentz"

# python main.py --gcn_flag --dataset="Citeseer" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-3 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GCNCiteser" \
#                 --num_workers 32 --verbose --patience 20 --model_save \

# python main.py --graphsage_flag --dataset="Citeseer" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-3 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GraphSageCiteseer" \
#                 --num_workers 32 --verbose --patience 20 --model_save \

# python main.py --gat_flag --dataset="Citeseer" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-3 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GATCiteseer" \
#                 --num_workers 32 --verbose --patience 20 --model_save \

# python main.py --gcn_flag --dataset="Citeseer" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-4 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GCNPoincareCiteseer" \
#                 --num_workers 32 --verbose --patience 20 --model_save \
#                 --hyperbolic_flag --hyperbolic_model "poincare"

# python main.py --gcn_flag --dataset="Citeseer" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-4 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GCNLorentzCiteseer" \
#                 --num_workers 32 --verbose --patience 20 --model_save \
#                 --hyperbolic_flag --hyperbolic_model "lorentz"

# python main.py --random_walk_flag --dataset "Cora" --device "cuda" \
#                --random_walk_model "DeepWalk" \
#                --embedding_dim 32 --mlp_hidden_dim 16 \
#                --walk_length 10 --num_walks 10 \
#                --num_epochs 10 --num_runs 2 --lr 0.01 --weight_decay 5e-4 --batch_size 32 \
#                --num_workers 4 --verbose \
#                --output_dir "output" --file_name "DeepWalkCora" 

# python main.py --random_walk_flag --dataset "Cora" --device "cuda" \
#                --random_walk_model "Node2Vec" \
#                --embedding_dim 32 --mlp_hidden_dim 16 \
#                --walk_length 10 --num_walks 10 \
#                --num_epochs 10 --num_runs 2 --lr 0.01 --weight_decay 5e-4 --batch_size 32 \
#                --num_workers 4 --verbose \
#                --output_dir "output" --file_name "Node2VecCora" 

# python main.py --random_walk_flag --dataset "Citeseer" --device "cuda" \
#                --random_walk_model "DeepWalk" \
#                --embedding_dim 32 --mlp_hidden_dim 16 \
#                --walk_length 10 --num_walks 10 \
#                --num_epochs 10 --num_runs 2 --lr 0.01 --weight_decay 5e-4 --batch_size 32 \
#                --num_workers 4 --verbose \
#                --output_dir "output" --file_name "DeepWalkCiteseer" 

# python main.py --random_walk_flag --dataset "Citeseer" --device "cuda" \
#                --random_walk_model "Node2Vec" \
#                --embedding_dim 32 --mlp_hidden_dim 16 \
#                --walk_length 10 --num_walks 10 \
#                --num_epochs 10 --num_runs 2 --lr 0.01 --weight_decay 5e-4 --batch_size 32 \
#                --num_workers 4 --verbose \
#                --output_dir "output" --file_name "Node2VecCiteseer" 
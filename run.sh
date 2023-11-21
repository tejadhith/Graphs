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

python main.py --gcn_flag --dataset="Cora" --device="cpu" \
                --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
                --num_epochs 100 --num_runs 10 --lr 2e-4 --weight_decay 5e-4 --batch_size 256 --batch_norm \
                --output_dir "output" --file_name "GCNLorentzCora" \
                --num_workers 32 --verbose --patience 20 --model_save \
                --hyperbolic_flag --hyperbolic_model "lorentz"

# python main.py --gcn_flag --dataset="Citeseer" --device="cpu" \
#                 --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
#                 --num_epochs 100 --num_runs 10 --lr 2e-3 --weight_decay 5e-4 --batch_size 256 --batch_norm \
#                 --output_dir "output" --file_name "GCNCora" \
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

python main.py --gcn_flag --dataset="Citeseer" --device="cpu" \
                --embedding_dim 32 --gnn_hidden_dim 64 --mlp_hidden_dim 16 \
                --num_epochs 100 --num_runs 10 --lr 2e-4 --weight_decay 5e-4 --batch_size 256 --batch_norm \
                --output_dir "output" --file_name "GCNLorentzCiteseer" \
                --num_workers 32 --verbose --patience 20 --model_save \
                --hyperbolic_flag --hyperbolic_model "lorentz"

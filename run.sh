python3 main.py \
 --gpu 0 \
 --local_num_layer 2 \
 --hidden 64 \
 --num_epoch 100 \
 --tribe_encoder_gnn gin \
 --lr 3e-3 \
 --weight_decay 5e-3 \
 --fusion_mode mlp \
 --path_x {path to node attribute file} \
 --path_y {path to node label file} \
 --path_tribe_files {path to tribe-graph files} \
 --path_tribe_order {path to tribe_graph order file}

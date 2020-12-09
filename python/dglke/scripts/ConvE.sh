# 8GPU
#sudo /home/ubuntu/anaconda3/envs/dgl/bin/python ke_test.py --model_name ConvE --dataset FB15k --batch_size 512 --neg_sample_size 128 --hidden_dim 200 --gamma 5 --lr 0.001 --max_step 12000 --log_interval 5000 --batch_size_eval 16 --regularization_coef 0 --test --gpu 0 1 2 3 4 5 6 7 --loss_genre BCE --rel_part --dropout_ratio 0.2 0.2 0.3 --num_proc 8 --mode fit --label_smooth 0.1 ;
#sudo /home/ubuntu/anaconda3/envs/dgl/bin/python ke_test.py --model_name ConvE --dataset wn18 --batch_size 512 --neg_sample_size 128 --hidden_dim 200 --gamma 5 --lr 0.001 --max_step 12000 --log_interval 5000 --batch_size_eval 16 --regularization_coef 0 --test --gpu 0 1 2 3 4 5 6 7 --loss_genre BCE --rel_part --dropout_ratio 0.2 0.2 0.3 --num_proc 8 --mode fit --label_smooth 0.1 ;

# 1GPU
sudo /home/ubuntu/anaconda3/envs/dgl/bin/python ../ke_test.py --model_name ConvE --dataset FB15k --batch_size 256 --neg_sample_size 64--hidden_dim 200 --gamma 5 --lr 0.01 --max_step 200000 --log_interval 1000 --batch_size_eval 16 --regularization_coef 1e-9 --test --gpu 0 --loss_genre BCE --dropout_ratio 0.2 0.2 0.3 --num_proc 1 --mode fit --label_smooth 0.1 --init_strat xavier ;
#sudo /home/ubuntu/anaconda3/envs/dgl/bin/python ke_test.py --model_name ConvE --dataset wn18 --batch_size 256 --neg_sample_size 128 --hidden_dim 200 --gamma 5 --lr 0.001 --max_step 200000 --log_interval 5000 --batch_size_eval 16  --regularization_coef 0 --test --gpu 0 --loss_genre BCE --dropout_ratio 0.2 0.2 0.3 --num_proc 1 --mode fit --label_smooth 0.1 ;

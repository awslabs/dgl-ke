#TransE_l2 8 GPU training
dglke_train --model_name TransE_l2 --dataset biokg --batch_size 512 --log_interval 100 \
--neg_sample_size 128 --regularization_coef=1e-9 --hidden_dim 2000 --gamma 20 -adv -a 1.0 \
--lr 0.25 --max_step 7500 --no_eval_filter --test --batch_size_eval 32 --async_update \
--gpu 0 1 2 3 4 5 6 7 --neg_sample_size_eval 500
################## Script Result #################
#training takes 219.76592564582825 seconds
#-------------- Test result --------------
#Test average MRR : 0.5833251626236052
#est average MR : 7.5597869466445635
#Test average HITS@1 : 0.4186774728310923
#Test average HITS@3 : 0.7121477251795911
#Test average HITS@10 : 0.8612267452569534
#-----------------------------------------
#testing takes 21.882 seconds
##################################################

#DistMult 8 GPU training
dglke_train --model_name DistMult --dataset biokg --batch_size 512 --log_interval 1000 \
--neg_sample_size 128 --hidden_dim 2000 --gamma 500 -adv -a 1.0 \
--lr 0.1 --max_step 75000 --no_eval_filter --test --batch_size_eval 32 --async_update \
--gpu 0 1 2 3 4 5 6 7 --neg_sample_size_eval 500 -rc 0.000002
################## Script Result #################
#training takes 1865.593612909317 seconds
#-------------- Test result --------------
#Test average MRR : 0.6357657854284788
#Test average MR : 7.190302081414625
#Test average HITS@1 : 0.48547307668692824
#Test average HITS@3 : 0.758267329772211
#Test average HITS@10 : 0.8787714127832014
#-----------------------------------------
#testing takes 21.496 seconds
##################################################


#RotatE 8 GPU training
dglke_train --model_name RotatE --dataset biokg --batch_size 512 --log_interval 1000 \
--neg_sample_size 128 --hidden_dim 1000 --gamma 20 -adv -a 1.0 \
--lr 0.01 --max_step 75000 --no_eval_filter --test --batch_size_eval 32 --async_update \
--gpu 0 1 2 3 4 5 6 7 --neg_sample_size_eval 500 -de
################## Script Result #################
#training takes 3053.7949199676514 seconds
#-------------- Test result --------------
#Test average MRR : 0.6298718902242
#Test average MR : 7.166878492048873
#Test average HITS@1 : 0.4744458770798797
#Test average HITS@3 : 0.7585313440167004
#Test average HITS@10 : 0.8799748265487812
#-----------------------------------------
#testing takes 23.890 seconds
##################################################


#ComplEx 8 GPU training
dglke_train --model_name ComplEx --dataset biokg --batch_size 512 --log_interval 1000 \
--neg_sample_size 128 --hidden_dim 1000 --gamma 500 -adv -a 1.0 \
--lr 0.1 --max_step 75000 --no_eval_filter --test --batch_size_eval 32 --async_update \
--gpu 0 1 2 3 4 5 6 7 --neg_sample_size_eval 500 -de -dr -rc 0.000002
################## Script Result #################
#training takes 1908.9747366905212 seconds
#-------------- Test result --------------
#Test average MRR : 0.6271215482568984
#Test average MR : 7.514815497022165
#Test average HITS@1 : 0.47368146374409037
#Test average HITS@3 : 0.7528458279609505
#Test average HITS@10 : 0.8740222263154663
#-----------------------------------------
#testing takes 21.795 seconds
##################################################
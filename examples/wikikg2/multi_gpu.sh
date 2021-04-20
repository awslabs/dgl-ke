#TransE_l2 8 GPU training
dglke_train --model_name TransE_l2 --dataset wikikg2 --batch_size 512 --log_interval 100 \
--neg_sample_size 128 --regularization_coef=1e-9 --hidden_dim 500 --gamma 30 -adv -a 1.0 \
--lr 0.25 --max_step 2500 --no_eval_filter --test --batch_size_eval 32 --async_update \
--gpu 0 1 2 3 4 5 6 7 --neg_sample_size_eval 500
################## Script Result #################
#training takes 63.87991690635681 seconds
#-------------- Test result --------------
#Test average MRR : 0.4243544729339413
#Test average MR : 83.01282280471078
#Test average HITS@1 : 0.3798816459302005
#Test average HITS@3 : 0.4367739661143811
#Test average HITS@10 : 0.4995881665978885
#-----------------------------------------
#testing takes 51.257 seconds
##################################################

#DistMult 8 GPU training
dglke_train --model_name DistMult --dataset wikikg2 --batch_size 512 --log_interval 100 \
--neg_sample_size 128 --hidden_dim 500 --gamma 500 -adv -a 1.0 \
--lr 0.1 --max_step 10000 --no_eval_filter --test --batch_size_eval 32 --async_update \
--gpu 0 1 2 3 4 5 6 7 --neg_sample_size_eval 500 -rc 0.000002
################## Script Result #################
#training takes 184.0140368938446 seconds
#-------------- Test result --------------
#Test average MRR : 0.3617282070612866
#Test average MR : 64.65302075205959
#Test average HITS@1 : 0.2815770963823819
#Test average HITS@3 : 0.39270528600284355
#Test average HITS@10 : 0.5065183286747986
#-----------------------------------------
#testing takes 48.500 seconds
##################################################

#RotatE 8 GPU training
dglke_train --model_name RotatE --dataset wikikg2 --batch_size 512 --log_interval 100 \
--neg_sample_size 128 --hidden_dim 250 --gamma 5 -adv -a 1.0 \
--lr 0.01 --max_step 8000 --no_eval_filter --test --batch_size_eval 32 --async_update \
--gpu 0 1 2 3 4 5 6 7 --neg_sample_size_eval 500 -de
################## Script Result #################
#training takes 179.34141492843628 seconds
#-------------- Test result --------------
#Test average MRR : 0.44640414701755626
#Test average MR : 58.77002738316211
#Test average HITS@1 : 0.3917220650813726
#Test average HITS@3 : 0.45981241113838106
#Test average HITS@10 : 0.5494851664792672
#-----------------------------------------
#testing takes 51.616 seconds
##################################################


#ComplEx 8 GPU training
dglke_train --model_name ComplEx --dataset wikikg2 --batch_size 512 --log_interval 1000 \
--neg_sample_size 128 --hidden_dim 250 --gamma 143 -adv -a 1.0 \
--lr 0.1 --max_step 10000 --no_eval_filter --test --batch_size_eval 32 --async_update \
--gpu 0 1 2 3 4 5 6 7 --neg_sample_size_eval 500 -de -dr
################## Script Result #################
#training takes 187.80023765563965 seconds
#-------------- Test result --------------
#Test average MRR : 0.4488316744799845
#Test average MR : 50.726986198151174
#Test average HITS@1 : 0.3775585045685941
#Test average HITS@3 : 0.47208888918590647
#Test average HITS@10 : 0.5845369505616138
#-----------------------------------------
#testing takes 51.129 seconds
##################################################

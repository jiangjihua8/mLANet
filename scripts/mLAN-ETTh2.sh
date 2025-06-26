#export CUDA_VISIBLE_DEVICES=1

model_name=mLAN
seq_len=720
convWindows='24 48 72 144' 

echo ETTh2
for pred_len in 96 192 336 720 
do
    echo grid-parameters: seq_len:$seq_len pred_len:$pred_len 
    python -u main.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv\
    --model_id ETTh2_multi_${seq_len}_${pred_len}\
    --model $model_name \
    --data ETTh2\
    --features M\
    --seq_len $seq_len\
    --label_len 48\
    --pred_len $pred_len\
    --des 'Exp'\
    --enc_in 7\
    --channels 8\
    --context_length 512\
    --conv1d_kernel_size 8\
    --qkv_proj_blocksize 4\
    --num_heads 4\
    --proj_factor 1.0\
    --bias True\
    --d_model 512\
    --inplanes 8\
    --ratio 0.25\
    --batch_size 256\
    --train_epochs 200\
    --patience 30\
    --learning_rate 0.0001\
    --itr 1 \
    --loss mae\
    --convWindows $convWindows --rnnMixTemperature 0.002\
    --lradj type3
done  > ./'mLAN-ETTh2'.log
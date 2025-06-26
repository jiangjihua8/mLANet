#export CUDA_VISIBLE_DEVICES=2

model_name=mLAN
seq_len=720
convWindows='24 48 72 96 144'

echo traffic

for pred_len in  96 192 336 720 
do
    echo grid-parameters: seq_len:$seq_len pred_len:$pred_len 
    python -u main.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 862 \
    --channels 8\
    --context_length 880\
    --conv1d_kernel_size 8\
    --qkv_proj_blocksize 4\
    --num_heads 4\
    --proj_factor 1.0\
    --bias True\
    --inplanes 8\
    --ratio 0.25\
    --des 'Exp' \
    --d_model 880\
    --batch_size 16\
    --learning_rate 0.0001\
    --itr 1 \
    --loss mae\
    --train_epochs 200\
    --patience 30\
    --convWindows $convWindows --rnnMixTemperature 0.002 --lradj type3
done  > ./'mLAN_Traffic'.log
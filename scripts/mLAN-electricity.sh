#export CUDA_VISIBLE_DEVICES=1

model_name=mLAN
seq_len=660
convWindows='22 44 66 88 132' # There are only 22 data records a day in electricity, missing data at 6, 7, 8 o 'clock

echo electricity
for pred_len in  96 192 336 720
do
    echo grid-parameters: seq_len:$seq_len pred_len:$pred_len 
    python -u main.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len ${seq_len} \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 321 \
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
    --batch_size 16\
    --learning_rate 0.0005\
    --itr 1 \
    --loss mae\
    --train_epochs 100\
    --patience 50\
    --convWindows $convWindows --rnnMixTemperature 0.002 --lradj type3
done  > ./'mLAN-Electricity'.log
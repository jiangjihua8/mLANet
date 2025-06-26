#export CUDA_VISIBLE_DEVICES=0

model_name=mLAN
seq_len=720
convWindows='6 24 48 144' 

echo weather
for pred_len in 96 192 336 720
do
    echo grid-parameters: pred_len:$pred_len  seq_len:$seq_len d_ff:$d_ff 
    python -u main.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_${seq_len}_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 21 \
      --des 'Exp' \
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
      --itr 1 \
      --loss mae\
      --train_epochs 200\
      --patience 30\
      --batch_size 64 --learning_rate 0.0001\
      --convWindows $convWindows --rnnMixTemperature 0.002 --lradj type3
done > ./'mLAN_Weather'.log
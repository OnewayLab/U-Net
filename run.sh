export CUDA_VISIBLE_DEVICES=0

python main.py \
    --data_path "./data/sidewalk-semantic" \
    --input_size 724 \
    --batch_size 16 \
    --total_steps 4000 \
    --learning_rate 0.0005 \
    --output_path "./output"

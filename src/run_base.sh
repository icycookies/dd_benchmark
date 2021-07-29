python3 train.py \
    --config-file fusion_net_xlnet_gru_att.json \
    --num-epochs 20 \
    --dropout 0.1 \
    --lr 0.0001 \
    --num-features 77 \
    --batch-size 4 \
    --max-seq-len 1024 \
    --gradient-accumulation-steps 16 \
    --device 3 \
    --eval-steps 200 \
    --eval-metric f1 precision recall accuracy \
    --save
python3 train.py \
    --config-file microblog_text_gru_att.json \
    --num-epochs 10 \
    --dropout 0.1 \
    --lr 0.0001 \
    --num-features 77 \
    --batch-size 4 \
    --max-seq-len 32 \
    --max-microblogs 64 \
    --gradient-accumulation-steps 16 \
    --device 2 \
    --logging-steps 20 \
    --eval-steps 400 \
    --eval-metric f1 precision recall accuracy \
    --save
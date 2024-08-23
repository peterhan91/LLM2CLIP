MODEL=EVA02-CLIP-L-14-336
PRETRAINED=eva_clip

# Following OpenCLIP, we preprocess data by webdataset. We concat paths of LAION-2B and COYO-700M with `;`.
# MERGE_2B_DATA_PATH="/path/to/laion2b_en_data/img_data/{000000..164090}.tar;/path/to/coyo700m_en_data/img_data/{000000..047435}.tar"
# LAION_2B_DATA_PATH="/path/to/laion2b_en_data/img_data/{000000..164090}.tar"
# VAL_DATA_PATH=/path/to/IN-1K/val

cd rei
python -m torch.distributed.launch --nproc_per_node=4 \
        --use_env training/main.py \
        --enable-deepspeed \
        --grad-checkpointing \
        --name="T_vit_1024x4_Vlr1e-5T1e-6_Tcc3m_1s0l_woWup-load_Rcc3m" \
        --save-frequency 1  \
        --zeroshot-frequency 1 \
        --report-to="tensorboard, wandb" \
        --wandb-project-name="eva_llm" \
        --wandb-notes="EVA02-CLIP-L-14-336" \
        --dataset-resampled \
        --train-data-file=training/tune_datasets.yaml \
        --eval-data-file=training/tune_eval_datasets.yaml \
        --imagenet-val=${HOME}/data/imagenet/val.zip \
        --imagenet-val-text=${HOME}/data/imagenet/val_map.txt \
        --pretrained=${PRETRAINED} \
        --precision "fp16" \
        --warmup 0 \
        --batch-size=1024 \
        --log-every-n-steps 10 \
        --epochs=10 \
        --lr=1e-5 \
        --visual-lr=1e-5 \
        --text-lr=1e-6 \
        --wd=0.05 \
        --visual-wd=0.05 \
        --text-wd=0.05 \
        --ld=1.0 \
        --text-ld=1.01 \
        --visual-ld=0.85 \
        --grad-clip-norm=5.0 \
        --smoothing=0. \
        --workers=8 \
        --model=${MODEL} \
        --seed 4096 \
        --gather-with-grad \
        --local-loss \
        --force-custom-clip \
        --optimizer="ap_adamw" \
        --zero-stage=1 \
        --dataset-type "json" 
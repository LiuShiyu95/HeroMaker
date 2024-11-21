GPUS=0

NAME=test
EXP_NAME=base

ROOT_DIRECTORY="all_sequences/$NAME/$NAME"
MODEL_SAVE_PATH="ckpts/all_sequences/$NAME"
LOG_SAVE_PATH="logs/all_sequences/$NAME"
MASK_DIRECTORY="all_sequences/$NAME/${NAME}_masks_0 all_sequences/$NAME/${NAME}_masks_1"

python train_deform.py --root_dir $ROOT_DIRECTORY \
                --model_save_path $MODEL_SAVE_PATH \
                --log_save_path $LOG_SAVE_PATH  \
                --mask_dir $MASK_DIRECTORY \
                --gpus $GPUS \
                --encode_w --annealed \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME}

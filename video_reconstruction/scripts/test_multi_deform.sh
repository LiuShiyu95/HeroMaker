GPUS=0

NAME=test
EXP_NAME=base

ROOT_DIRECTORY="all_sequences/$NAME/$NAME"
LOG_SAVE_PATH="logs/test_all_sequences/$NAME"
MASK_DIRECTORY="all_sequences/$NAME/${NAME}_masks_0 all_sequences/$NAME/${NAME}_masks_1"
WEIGHT_PATH=ckpts/all_sequences/$NAME/${EXP_NAME}/step=15000.ckpt

python train_deform.py --test --encode_w \
                --root_dir $ROOT_DIRECTORY \
                --log_save_path $LOG_SAVE_PATH \
                --mask_dir $MASK_DIRECTORY \
                --weight_path $WEIGHT_PATH \
                --gpus $GPUS \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME} \
                --save_deform False
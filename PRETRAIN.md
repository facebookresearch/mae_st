## Pre-training MAE
To pre-train ViT-Large (recommended default), run the following:

```
python run_pretrain.py \
    --path_to_data_dir ${KINETICS_DIR} \
    --batch_size 2 \
    --model mae_vit_large_patch16 \
    --no_env \
    --epochs 100 \
    --distributed \
    --num_frames 16 \
    --decoder_embed_dim 512 \
    --decoder_depth 4 \
    --pin_mem \
    --num_workers 14 \
    --t_patch_size 2 \
    --repeat_aug 4 \
    --sampling_rate 4 \
    --norm_pix_loss \
    --blr 1.6e-3 \
    --warmup_epochs 5 \
    --mask_ratio 0.9 \
    --pred_t_dim 8 \
    --clip_grad 0.02 \
```

blr is the base learning rate. The actual lr is computed by the linear scaling rule: lr = blr * effective batch size / 256.
Here we use --norm_pix_loss as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based constructiomae_vit_large_patch16n and turn off --norm_pix_loss.
To train ViT-Base or ViT-Huge, set --model mae_vit_base_patch16 or --model mae_vit_huge_patch14.

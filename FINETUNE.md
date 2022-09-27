## Fine-tuning Pre-trained MAE for Classification

### Evaluation

As a sanity check, run evaluation using our ImageNet **fine-tuned** models:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>

<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint on Kinetics-400</td>

<td align="center"><a href="https://dl.fbaipublicfiles.com/mae_st/pretrain/vit_l/k400_400ep_rep4_mr9_16x4.pyth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae_st/pretrain/vit_h/k400_400ep_rep4_mr9_16x4.pyth">download</a></td>

</tr>
<tr><td align="left">md5</td>

<td align="center"><tt>edf3a5</tt></td>
<td align="center"><tt>3d7f64</tt></td>
</tr>
</tbody></table>

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>


<tr><td align="left">pre-trained checkpoint on Kinetics-600</td>


<td align="center"><a href="https://dl.fbaipublicfiles.com/mae_st/pretrain/vit_l/k600_400ep_rep4_mr9_16x4.pyth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae_st/pretrain/vit_h/k600_400ep_rep4_mr9_16x4.pyth">download</a></td>

</tr>
<tr><td align="left">md5</td>

<td align="center"><tt>9a9645</tt></td>
<td align="center"><tt>27495e</tt></td>
</tr>
</tbody></table>

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>


<tr><td align="left">pre-trained checkpoint on Kinetics-700</td>

<td align="center"><a href="https://dl.fbaipublicfiles.com/mae_st/pretrain/vit_l/k700_400ep_rep4_mr9_16x4.pyth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae_st/pretrain/vit_h/k700_400ep_rep4_mr9_16x4.pyth">download</a></td>

</tr>
<tr><td align="left">md5</td>

<td align="center"><tt>cdbada</tt></td>
<td align="center"><tt>4c4e3c</tt></td>
</tr>
</tbody></table>


Evaluate ViT-Large: (`${KINETICS_DIR}` is a directory containing `{train, val}` sets of Kinetics):
```
python run_finetune.py --path_to_data_dir ${KINETICS_DIR} --rand_aug --epochs 50 --repeat_aug 2 --model vit_large_patch16 --batch_size 2 --distributed --dist_eval --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --mixup_prob 1.0 --blr 0.0024 --num_frames 16 --sampling_rate 4 --dropout 0.3 --warmup_epochs 5 --layer_decay 0.75 --drop_path_rate 0.2 --aa rand-m7-mstd0.5-inc1 --clip_grad 5.0 --fp32"}${FINETUNE_APPENDIX}
```
This should give:
```
* Acc@1 84.35
```

#### Notes

- The pre-trained models we provide are trained with *normalized* pixels `--norm_pix_loss` (1600 effective epochs). The models are pretrained in PySlowFast codebase.

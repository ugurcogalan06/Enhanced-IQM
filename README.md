This is the official repository for the implementation of the paper: Enhancing image quality prediction with self-supervised visual masking
You can find the paper here: https://arxiv.org/abs/2305.19858

The code for the enhanced metric for MAE can be run simply with:
```bash
python Masked_L1.py --ref images/ref.BMP --dist images/dist.BMP
```

The code was tested under Debian GNU/Linux 11.

Dependencies:

pytorch-cuda==11.7

numpy==1.23.3

torchvision==0.14.0

pillow==9.2.0

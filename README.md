This is the official repository for the implementation of the paper: Enhancing image quality prediction with self-supervised visual masking
You can find the paper here: https://arxiv.org/abs/2305.19858

The code can be run simply with:
python Masked_L1.py --ref images/ref.BMP --dist images/dist.BMP

Dependencies:
pytorch-cuda==11.7
numpy==1.23.3
torchvision==0.14.0
pillow==9.2.0

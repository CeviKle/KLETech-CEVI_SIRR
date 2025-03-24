## Requirements
We recommend torch 2.x for our code, but it should works fine with most of the modern versions.

```
pip install torch>=2.0 torchvision
pip install einops ema-pytorch fsspec fvcore huggingface-hub matplotlib numpy opencv-python omegaconf pytorch-msssim scikit-image scikit-learn scipy tensorboard tensorboardx wandb timm
```

# Testing 
Please download the pretrained weights from [this google drive](https://drive.google.com/drive/folders/1xSm7Pm1aIAHKqI8h_rp3UYucNM1TcGEh?usp=sharing). Please put the cls_model.pth under "pretrained" folder.

```python
python3 test_sirs.py --icnn_path path/to/result.pth --resume
```

# Results
The results are stored in 
```
./results/real20
```

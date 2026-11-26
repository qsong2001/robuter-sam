# Dataset layout

Put your own RGB images and binary masks in separate splits:

```text
datasets/my_dataset/
├── train/
│   ├── images/scene001.jpg
│   └── masks/scene001.png
└── val/
    ├── images/scene101.jpg
    └── masks/scene101.png
```

Match filenames by stem, including any nested folders. Each mask must be
single-channel, have the same height and width as its image, and mark
background with zero and the target foreground with a nonzero value.
Use one target object per mask for box-prompted training. Split images before
generating the manifests; do not reuse training images for validation.

See [the example manifest](example.jsonl). Its images are not shipped.

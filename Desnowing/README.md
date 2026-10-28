# FocalNet Desnowing — Combined Dataset Training

The loader combines CSD, SRRS, and Snow100K into one training stream using balanced dataset-level sampling by default.

## Folder structure

Place `dataset` in the FocalNet root, next to `Desnowing`:

```text
FocalNet/
├── dataset/
│   ├── CSD/
│   │   ├── train2500/
│   │   │   ├── Snow/
│   │   │   └── Gt/
│   │   └── test2000/
│   │       ├── Snow/
│   │       └── Gt/
│   ├── SRRS/       # same structure
│   └── Snow100K/   # same structure
└── Desnowing/
    ├── main.py
    ├── train.py
    ├── valid.py
    ├── eval.py
    ├── check_pipeline.py
    ├── data/
    └── models/
```

Snowy inputs go in `Snow/`; clean targets go in `Gt/`. Image pairs must have matching filenames (or filename stems) and the same dimensions.

## Train one model on all three datasets

The code automatically uses `FocalNet/dataset` by default, so from the `Desnowing` folder run:

```powershell
python main.py --mode train --batch-size 4 --num-epoch 2000
```

Override the path only if your dataset folder is elsewhere:

```powershell
python main.py --mode train --data-dir "D:\datasets" --batch-size 4 --num-epoch 2000
```

## Resume training

```powershell
python main.py --mode train --resume "results\FocalNet\Desnowing\model.pkl" --batch-size 4 --num-epoch 2000
```

`--num-epoch` is the total target epoch. `model.pkl` stores model weights, optimizer state, completed epoch, and best validation PSNR.

## Validation and final evaluation

If a dataset has no `val`, `valid`, or `validation` folder, the loader creates a deterministic 90/10 split from `train2500` for training and validation. `test2000` is reserved for final evaluation.

Evaluate each dataset separately:

```powershell
python main.py --mode test --data-dir "..\dataset\CSD" --test-model "results\FocalNet\Desnowing\Best.pkl"
python main.py --mode test --data-dir "..\dataset\SRRS" --test-model "results\FocalNet\Desnowing\Best.pkl"
python main.py --mode test --data-dir "..\dataset\Snow100K" --test-model "results\FocalNet\Desnowing\Best.pkl"
```

## Check the pipeline

```powershell
python check_pipeline.py
```

The checker uses the same default dataset path. You can override it with `--data-dir`.

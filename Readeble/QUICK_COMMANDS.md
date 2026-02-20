# Quick Command Reference

## Verify Data Loading
```powershell
python quick_test.py
```

## Train Model (Default Settings)
```powershell
python train_iclr.py
```

## Train Model (Custom Settings)
```powershell
# Small/fast model for testing
python train_iclr.py `
    --base_model "distilbert-base-uncased" `
    --batch_size 4 `
    --num_epochs 5

# Full model for production
python train_iclr.py `
    --base_model "allenai/longformer-base-4096" `
    --batch_size 2 `
    --learning_rate 2e-5 `
    --num_epochs 20 `
    --output_dir "./outputs_production"
```

## Monitor Training
```powershell
tensorboard --logdir ./logs
```
Then open: http://localhost:6006

## All Arguments
```powershell
python train_iclr.py --help
```

Available arguments:
- `--data_path`: Path to data folder (default: C:/Facultate/Licenta/data)
- `--output_dir`: Output directory (default: ./outputs)
- `--base_model`: Transformer model (default: allenai/longformer-base-4096)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--num_epochs`: Number of epochs (default: 20)


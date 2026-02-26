# Console & File Logging Guide

## Alternative Ways to View Training Logs (Without TensorBoard)

Since you're experiencing issues with TensorBoard and `pkg_resources`, here are multiple ways to monitor your training:

---

## ✅ 1. Console Output (Already Implemented)

The training script already prints comprehensive logs to the console:

```bash
python train.py
```

**What you'll see:**
```
================================================================================
Epoch 1/20
================================================================================
🔒 Backbone frozen
Epoch 1: 100%|████████| 64/64 [02:15<00:00, 2.1s/it, loss=0.452]

Train Loss: 0.4523

Evaluating on dev set...

Dev Set Metrics
================================================================================

Macro Averages:
  accuracy: 0.6250
  mae: 0.5125
  mse: 0.3421
  qwk: 0.4156      ← Primary metric
  rmse: 0.5849
  spearman: 0.7234

Per-Dimension Metrics:

  IMPACT:
    accuracy: 0.7500
    mae: 0.3750
    qwk: 0.5234
    ...
```

**Redirect to file:**
```bash
python train.py 2>&1 | Tee-Object -FilePath training_log.txt
```

---

## ✅ 2. CSV Logger (Add This Code)

Create a simple CSV logger that saves metrics every epoch:

### Add to `trainer.py` after line 237 (in the `train` method):

```python
# After: self.logger.add_scalar('dev/avg_qwk', dev_metrics['avg_qwk'], epoch)

# Save metrics to CSV
import csv
csv_path = os.path.join(self.config.output_dir, 'training_metrics.csv')
csv_exists = os.path.exists(csv_path)

with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'dev_qwk', 'dev_accuracy', 'dev_mae', 'dev_spearman', 'lr'])
    if not csv_exists:
        writer.writeheader()
    
    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
    writer.writerow({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'dev_qwk': dev_metrics['avg_qwk'],
        'dev_accuracy': dev_metrics['macro_avg']['accuracy'],
        'dev_mae': dev_metrics['macro_avg']['mae'],
        'dev_spearman': dev_metrics['macro_avg']['spearman'],
        'lr': lr
    })
```

**Then open in Excel or view in terminal:**
```bash
# PowerShell
Get-Content outputs/training_metrics.csv | ConvertFrom-Csv | Format-Table

# Or import to pandas
python -c "import pandas as pd; df = pd.read_csv('outputs/training_metrics.csv'); print(df)"
```

---

## ✅ 3. JSON Logger (More Detailed)

Save all metrics as JSON for later analysis:

### Create `C:\Facultate\Licenta\json_logger.py`:

```python
import json
import os
from datetime import datetime

class JSONLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.history = {
            'config': {},
            'epochs': []
        }
    
    def log_config(self, config_dict):
        self.history['config'] = config_dict
    
    def log_epoch(self, epoch, train_loss, dev_metrics, lr=None):
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'dev_metrics': dev_metrics,
            'learning_rate': lr
        }
        self.history['epochs'].append(epoch_data)
        
        # Save incrementally
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training log saved to {self.log_path}")
```

### Use in `train.py`:

```python
from json_logger import JSONLogger

# After creating trainer:
json_logger = JSONLogger(training_config.output_dir)
json_logger.log_config({
    'model': model_config.__dict__,
    'training': training_config.__dict__,
    'data': data_config.__dict__
})

# In training loop (after dev evaluation):
json_logger.log_epoch(
    epoch=epoch,
    train_loss=train_loss,
    dev_metrics=dev_metrics,
    lr=scheduler.get_last_lr()[0] if scheduler else training_config.learning_rate
)
```

**View the JSON:**
```bash
# Pretty print in terminal
python -c "import json; print(json.dumps(json.load(open('outputs/training_log_*.json')), indent=2))"

# Or use any JSON viewer
```

---

## ✅ 4. Plot Metrics with Matplotlib

### Create `C:\Facultate\Licenta\plot_training.py`:

```python
import json
import matplotlib.pyplot as plt
import glob
import os

def plot_training_curves(log_path=None):
    """Plot training curves from JSON log."""
    
    if log_path is None:
        # Find most recent log
        log_files = glob.glob('outputs/training_log_*.json')
        if not log_files:
            print("No training logs found!")
            return
        log_path = max(log_files, key=os.path.getctime)
    
    print(f"Loading: {log_path}")
    
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    epochs = [e['epoch'] for e in data['epochs']]
    train_loss = [e['train_loss'] for e in data['epochs']]
    dev_qwk = [e['dev_metrics']['avg_qwk'] for e in data['epochs']]
    dev_accuracy = [e['dev_metrics']['macro_avg']['accuracy'] for e in data['epochs']]
    dev_mae = [e['dev_metrics']['macro_avg']['mae'] for e in data['epochs']]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Train Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', marker='o')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Dev QWK (Primary Metric)
    axes[0, 1].plot(epochs, dev_qwk, 'g-', marker='o')
    axes[0, 1].set_title('Dev QWK (Primary Metric)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('QWK')
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=max(dev_qwk), color='r', linestyle='--', label=f'Best: {max(dev_qwk):.4f}')
    axes[0, 1].legend()
    
    # Dev Accuracy
    axes[1, 0].plot(epochs, dev_accuracy, 'orange', marker='o')
    axes[1, 0].set_title('Dev Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True)
    
    # Dev MAE
    axes[1, 1].plot(epochs, dev_mae, 'r-', marker='o')
    axes[1, 1].set_title('Dev MAE (Lower is Better)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = log_path.replace('.json', '_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Plot saved to {plot_path}")
    
    plt.show()

if __name__ == '__main__':
    plot_training_curves()
```

**Usage:**
```bash
# After training completes:
python plot_training.py
```

---

## ✅ 5. Live Progress with Rich (Fancy Console)

Install Rich library:
```bash
pip install rich
```

### Create enhanced progress display:

```python
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

# In training loop:
table = Table(title=f"Epoch {epoch+1}/{num_epochs}")
table.add_column("Metric", style="cyan")
table.add_column("Value", style="magenta")

table.add_row("Train Loss", f"{train_loss:.4f}")
table.add_row("Dev QWK", f"{dev_metrics['avg_qwk']:.4f}")
table.add_row("Dev Accuracy", f"{dev_metrics['macro_avg']['accuracy']:.4f}")
table.add_row("Dev MAE", f"{dev_metrics['macro_avg']['mae']:.4f}")
table.add_row("Learning Rate", f"{lr:.2e}")

console.print(table)
```

---

## ✅ 6. Simple Text File Logger

Most minimal approach - just append metrics to a text file:

```python
# In trainer.py train() method:
with open(os.path.join(self.config.output_dir, 'training.log'), 'a') as f:
    f.write(f"Epoch {epoch+1}: loss={train_loss:.4f}, qwk={dev_metrics['avg_qwk']:.4f}, "
            f"acc={dev_metrics['macro_avg']['accuracy']:.4f}, "
            f"mae={dev_metrics['macro_avg']['mae']:.4f}\n")
```

**View the log:**
```bash
cat outputs/training.log
# or
Get-Content outputs/training.log
```

---

## 🎯 Recommended Setup

**For your current situation, I recommend:**

1. **Use the console output** (already works!)
2. **Add CSV logger** (5 lines of code, easy to view in Excel)
3. **Create plot_training.py** (visualize after training)

This gives you:
- Real-time feedback (console)
- Structured data (CSV)
- Visual analysis (plots)

**No TensorBoard needed!**

---

## 📊 Monitoring During Training

### What to Watch:

1. **Train Loss** - Should decrease steadily
   - If increasing: LR too high
   - If flat: Model saturated or collapsed

2. **Dev QWK** - Primary metric (should increase)
   - Target: >0.3 for 80 samples
   - >0.5 is very good
   - >0.7 is excellent

3. **Dev MAE** - Should decrease (ideally <0.5)

4. **Dev Spearman** - Correlation (should be >0.6)

### Early Stopping Indicators:

- ✅ **Good**: Train loss ↓, Dev QWK ↑
- ⚠️ **Overfitting**: Train loss ↓, Dev QWK stable/↓
- 🛑 **Collapsed**: All predictions same class (QWK = 0)

---

## 🐛 Fixing TensorBoard (Optional)

If you still want TensorBoard working:

### Option 1: Reinstall setuptools properly
```bash
pip uninstall setuptools -y
pip install "setuptools<70" --force-reinstall
pip install tensorboard --force-reinstall
```

### Option 2: Use older TensorBoard
```bash
pip install "tensorboard<2.18"
```

### Option 3: Downgrade Python
```bash
# TensorBoard works better with Python 3.10-3.11
# Create new venv with older Python
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements.txt
```

---

## 📁 Output Files

After training, you'll have:

```
outputs/
├── best_model.pt              # Best model checkpoint
├── test_results.pt            # Final test metrics
├── training_metrics.csv       # CSV with epoch metrics
├── training_log_*.json        # Detailed JSON log
├── training_log_*_plot.png    # Training curves plot
└── training.log               # Simple text log

logs/
└── events.out.tfevents.*      # TensorBoard events (if working)
```

---

## 🚀 Quick Start

1. **Train with console logging:**
   ```bash
   python train.py
   ```

2. **Save console output to file:**
   ```bash
   python train.py 2>&1 | Tee-Object -FilePath training_output.txt
   ```

3. **View metrics after training:**
   ```bash
   # If you added CSV logger:
   Import-Csv outputs/training_metrics.csv | Format-Table
   
   # If you added plot script:
   python plot_training.py
   ```

That's it! No TensorBoard needed. 📊


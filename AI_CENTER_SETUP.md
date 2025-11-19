# Setup dan Running di AI Center UB DGX UB

## 🚀 Quick Start (Setelah Pull)

### Step 1: Copy Configuration File

```bash
cp config_aicenter.yaml.example config.yaml
```

Atau edit `config.yaml` langsung dan pastikan:
- `device_id: "aicenter"`
- `total_devices: 2` (jika pakai multi-device dengan laptop)
- `n_jobs: 60`
- `parallel_methods: true`

### Step 2: Install Dependencies (jika belum)

```bash
pip install -r requirements.txt
```

### Step 3: Download Data Solomon (jika belum ada)

```bash
python download_solomon.py
```

Atau pastikan data sudah ada di `data/solomon/`:
- C1/, C2/, R1/, R2/, RC1/, RC2/

### Step 4: Setup Multi-Device (jika pakai 2 devices)

Jika menggunakan multi-device dengan laptop, jalankan setup:

```bash
python setup_2_devices.py
```

Ini akan:
- Assign instances ke masing-masing device
- Buat device config files
- Pastikan tidak ada overlap

**Atau** jika sudah setup sebelumnya, pastikan file `results/device_aicenter_config.json` ada.

### Step 5: Verifikasi Configuration

Cek config yang akan digunakan:

```bash
# Lihat config
cat config.yaml | grep -A 10 "evaluation:"

# Pastikan:
# - parallel: true
# - n_jobs: 60
# - parallel_methods: true
```

### Step 6: Jalankan Experiment

```bash
python main.py --config config.yaml
```

## 📊 Monitoring Progress

### Check Progress
```bash
# Lihat checkpoint
cat results/checkpoint.json | python -m json.tool

# Atau gunakan script
python view_results.py C101
```

### Monitor Logs
Program akan print progress secara real-time:
- Progress setiap 10 detik
- Generation progress dari GA
- Results per method

## ⚙️ Configuration Details

### AI Center Optimal Settings

```yaml
evaluation:
  parallel: true
  n_jobs: 60              # 60 workers (256 cores - leave some for OS)
  parallel_methods: true  # CRITICAL: Enable parallel methods execution
```

**Kenapa 60 workers?**
- 256 cores total
- Leave ~196 cores untuk OS, system processes, overhead
- 60 workers optimal untuk balance performance dan stability

**Kenapa `parallel_methods: true`?**
- Semua tasks (methods × runs) dieksekusi dalam satu pool
- 60 tasks bisa berjalan bersamaan (mix methods dan runs)
- **Jauh lebih cepat** daripada sequential methods!

## 🔍 Troubleshooting

### Problem: "No instances found"
**Solution**: Pastikan data Solomon sudah didownload:
```bash
python download_solomon.py
```

### Problem: "Device assignment not found"
**Solution**: Jalankan setup:
```bash
python setup_2_devices.py
```

### Problem: "Permission denied" saat save checkpoint
**Solution**: Normal, sistem akan retry. Pastikan folder `results/` writable.

### Problem: Memory issues
**Solution**: Kurangi `n_jobs` jika perlu (misal 40-50).

## 📈 Expected Performance

Dengan 60 workers dan `parallel_methods: true`:
- **Speedup**: ~60x untuk multiple methods
- **Efficiency**: Semua cores digunakan optimal
- **Time**: Significantly faster than sequential

## 🎯 Next Steps

Setelah running:
1. Monitor progress dengan `view_results.py`
2. Check checkpoint untuk melihat completed methods
3. Results akan tersimpan di `results/experiments/`
4. Convergence history di `results/convergence/`

## 💡 Tips

1. **Run in screen/tmux** untuk long-running jobs:
   ```bash
   screen -S ga_experiment
   python main.py --config config.yaml
   # Ctrl+A, D untuk detach
   ```

2. **Monitor CPU usage**:
   ```bash
   htop
   # atau
   top
   ```

3. **Check disk space** (results bisa besar):
   ```bash
   df -h
   ```


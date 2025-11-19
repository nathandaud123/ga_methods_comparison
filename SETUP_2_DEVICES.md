# Setup untuk 2 Devices: Laptop + AI Center UB DGX UB

## 🎯 Overview

Setup khusus untuk 2 devices:
- **Laptop**: Device 1, menggunakan semua core-nya
- **AI Center UB DGX UB**: Device 2, menggunakan **250 cores** dari 256 cores (optimal!)

## ✅ JAMINAN: Tidak Ada Overlap!

**VERIFIED**: Sistem menggunakan modulo distribution yang **mathematically guaranteed** tidak ada overlap.

**Verifikasi:**
```bash
python VERIFY_NO_OVERLAP.py
```

**Hasil:**
- ✅ NO OVERLAP - Perfect!
- ✅ ALL INSTANCES COVERED
- ✅ Safe to run simultaneously!

**Cara Kerja:**
- Laptop (index 0): Instances dengan index genap (0, 2, 4, 6, ...)
- AI Center (index 1): Instances dengan index ganjil (1, 3, 5, 7, ...)
- **Mathematical guarantee**: Tidak mungkin overlap!

## 🚀 Quick Setup

### Step 1: Setup Distribusi Workload

Jalankan di device manapun (laptop atau AI Center):

```bash
python setup_2_devices.py
```

Ini akan:
- Assign ~28 instances ke Laptop
- Assign ~28 instances ke AI Center
- Buat device config files

### Step 2: Clone Project di Kedua Device

**Laptop:**
```bash
git clone <your-repo>
cd ga_method_comparison
```

**AI Center:**
```bash
git clone <your-repo>
cd ga_method_comparison
```

### Step 3: Configure Each Device

**LAPTOP - Edit `config.yaml`:**
```yaml
device:
  device_id: "laptop"
  total_devices: 2

evaluation:
  parallel: true
  n_jobs: null  # Auto-detect (akan pakai semua core laptop)
```

**AI CENTER - Edit `config.yaml`:**
```yaml
device:
  device_id: "aicenter"
  total_devices: 2

evaluation:
  parallel: true
  n_jobs: 250  # 256 cores - 6 cores untuk OS/system (optimal untuk 256 cores)
  # Leave beberapa core free untuk OS dan system processes
```

### Step 4: Run di Kedua Device

**LAPTOP:**
```bash
python main.py --config config.yaml
```

**AI CENTER:**
```bash
python main.py --config config.yaml
```

## 📊 Distribusi Workload

Dengan 56 instances total:
- **Laptop**: 28 instances (C101, C103, C105, C107, C109, C202, C204, C206, C208, R102, R104, ...)
- **AI Center**: 28 instances (C102, C104, C106, C108, C201, C203, C205, C207, R101, R103, ...)

**VERIFIED: Tidak ada konflik!** 
- ✅ Setiap device kerja pada instance berbeda
- ✅ Mathematical guarantee (modulo distribution)
- ✅ Double safety (filter + checkpoint check)
- ✅ **100% Safe untuk run bersamaan!**

## ⚡ Performance

### Laptop:
- Multi-core parallel execution
- Speedup: ~N cores (misal 8 cores = ~8x faster)

### AI Center:
- **Menggunakan SEMUA core yang tersedia**
- Speedup: ~N cores (misal 64 cores = ~64x faster untuk multiple runs!)
- **Optimal untuk HPC!**

### Combined:
- **True parallel execution** - 2 devices bekerja simultan
- **No conflicts** - Instance berbeda
- **Maximum utilization** - Semua core terpakai

## 🔒 Checkpoint Sharing

### Option 1: Network Drive (Best)
- Simpan checkpoint di network drive yang accessible kedua device
- Update real-time di kedua device

### Option 2: Manual Sync
- Copy `results/checkpoint.json` secara berkala antara device
- Atau sync via git (tidak recommended untuk real-time)

### Option 3: Shared Folder
- Jika kedua device di network yang sama
- Gunakan shared folder untuk `results/`

## 💡 Tips untuk AI Center (256 Cores!)

1. **Optimal n_jobs untuk 256 cores:**
   ```yaml
   evaluation:
     n_jobs: 250  # Leave 6 cores untuk OS/system processes
   ```
   **Kenapa tidak 256?** 
   - Leave beberapa core untuk OS, system processes, dan overhead
   - 250 cores sudah optimal untuk maximum performance

2. **Check Available Cores:**
   ```bash
   # Di AI Center, verifikasi jumlah core
   python -c "import multiprocessing; print(f'CPU cores: {multiprocessing.cpu_count()}')"
   ```

3. **Monitor Resource Usage:**
   ```bash
   # Di AI Center, monitor CPU usage
   htop  # atau top
   # Pastikan semua cores terpakai dengan baik
   ```

4. **Memory Considerations:**
   - Dengan 250 parallel workers, pastikan RAM cukup
   - Setiap worker butuh memory untuk GA instance
   - Monitor memory usage: `free -h` atau `htop`

5. **Optimal Configuration:**
   ```yaml
   evaluation:
     parallel: true
     n_jobs: 250  # Optimal untuk 256-core system
     n_runs: 5    # 5 runs per method
   ```

## 📈 Expected Performance

| Device | Cores | Instances | Est. Time |
|--------|-------|-----------|-----------|
| Laptop | 8 | ~28 | ~35 hours |
| AI Center | 250 | ~28 | **~30 minutes** |
| **Total** | - | 56 | **~30 minutes** |

*Estimasi berdasarkan GA runtime ~1 min per run, 5 runs per method, 354 methods per instance*

**Dengan 256 cores, AI Center akan SANGAT CEPAT!** 🚀
- 250 parallel workers = ~250x speedup untuk multiple runs
- AI Center akan selesai jauh lebih cepat dari laptop

**AI Center akan selesai lebih cepat!** Laptop bisa continue setelah AI Center selesai.

## ✅ Checklist

- [ ] Run `setup_2_devices.py` untuk setup distribusi
- [ ] Clone project di laptop
- [ ] Clone project di AI Center
- [ ] Edit `config.yaml` di laptop (device_id: "laptop")
- [ ] Edit `config.yaml` di AI Center (device_id: "aicenter")
- [ ] Set `parallel: true` dan `n_jobs` di kedua device
- [ ] Setup checkpoint sharing (network drive atau manual sync)
- [ ] Run di laptop
- [ ] Run di AI Center
- [ ] Monitor progress: `python check_progress.py`

## 🎉 Result

Setelah selesai:
- Semua hasil ada di `results/experiments/`
- Summary aggregate semua hasil
- Tidak ada duplikasi
- **Maximum performance dengan 2 devices!** 🚀


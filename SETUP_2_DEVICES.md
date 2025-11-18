# Setup untuk 2 Devices: Laptop + AI Center UB DGX UB

## 🎯 Overview

Setup khusus untuk 2 devices:
- **Laptop**: Device 1, menggunakan semua core-nya
- **AI Center UB DGX UB**: Device 2, menggunakan **SEMUA** core-nya (banyak core!)

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
  n_jobs: null  # Auto-detect (akan pakai SEMUA core AI Center!)
  # Atau set manual jika tahu jumlah core:
  # n_jobs: 64  # Contoh: jika ada 64 cores
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
- **Laptop**: ~28 instances (C101, C103, C105, R101, R103, ...)
- **AI Center**: ~28 instances (C102, C104, C106, R102, R104, ...)

**Tidak ada konflik!** Setiap device kerja pada instance berbeda.

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

## 💡 Tips untuk AI Center

1. **Check Available Cores:**
   ```bash
   # Di AI Center, cek jumlah core
   python -c "import multiprocessing; print(f'CPU cores: {multiprocessing.cpu_count()}')"
   ```

2. **Set n_jobs Manual:**
   ```yaml
   evaluation:
     n_jobs: 64  # Set sesuai jumlah core AI Center
   ```

3. **Monitor Resource Usage:**
   ```bash
   # Di AI Center, monitor CPU usage
   htop  # atau top
   ```

4. **Leave Some Cores Free:**
   ```yaml
   evaluation:
     n_jobs: 60  # Jika ada 64 cores, leave 4 free untuk OS
   ```

## 📈 Expected Performance

| Device | Cores | Instances | Est. Time |
|--------|-------|-----------|-----------|
| Laptop | 8 | ~28 | ~35 hours |
| AI Center | 64 | ~28 | ~4 hours |
| **Total** | - | 56 | **~4 hours** |

*Estimasi berdasarkan GA runtime ~1 min per run, 5 runs per method, 354 methods per instance*

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


# Multi-Device Execution Setup

## 🚀 Parallel Execution Across Multiple Devices

Sistem mendukung **distribusi workload** ke multiple devices untuk eksekusi paralel yang benar-benar terdistribusi!

## ✅ Keuntungan

- ✅ **True Parallelism** - Setiap device bekerja pada instance/method berbeda
- ✅ **No Conflicts** - Tidak ada race condition atau duplikasi kerja
- ✅ **Automatic Distribution** - Workload otomatis terbagi
- ✅ **Shared Checkpoint** - Progress ter-track di semua device
- ✅ **Flexible** - Bisa tambah/kurang device kapan saja

## 📋 Setup Steps

### Step 1: Setup Device Configurations

Jalankan script setup untuk membuat konfigurasi semua device:

```bash
python setup_multi_device.py --total-devices 3 --method modulo
```

**Options:**
- `--total-devices N`: Jumlah total device (e.g., 3)
- `--method modulo`: Distribusi modulo (round-robin)
- `--method type`: Distribusi berdasarkan tipe (C, R, RC)

**Output:**
- File `results/device_device1_config.json`
- File `results/device_device2_config.json`
- File `results/device_device3_config.json`
- dll.

### Step 2: Copy Project ke Setiap Device

**Option A: Git Clone (Recommended)**
```bash
# Di device 1
git clone <your-repo>
cd ga_method_comparison

# Di device 2
git clone <your-repo>
cd ga_method_comparison

# Di device 3
git clone <your-repo>
cd ga_method_comparison
```

**Option B: Copy Folder**
- Copy seluruh folder `ga_method_comparison` ke setiap device
- Pastikan semua file ter-copy dengan benar

### Step 3: Configure Each Device

Edit `config.yaml` di setiap device:

**Device 1:**
```yaml
device:
  device_id: "device1"
  total_devices: 3
```

**Device 2:**
```yaml
device:
  device_id: "device2"
  total_devices: 3
```

**Device 3:**
```yaml
device:
  device_id: "device3"
  total_devices: 3
```

### Step 4: Copy Device Config Files (Optional)

**Option A: Auto-generate (Recommended)**
- Biarkan auto-generate saat pertama kali run
- Sistem akan otomatis assign instances berdasarkan device_id

**Option B: Manual Copy**
- Copy file `results/device_deviceN_config.json` ke setiap device yang sesuai
- Atau run setup script di setiap device dengan device_id yang sesuai

### Step 5: Run on Each Device

**Device 1:**
```bash
python main.py --config config.yaml
```

**Device 2:**
```bash
python main.py --config config.yaml
```

**Device 3:**
```bash
python main.py --config config.yaml
```

## 📊 Distribution Methods

### 1. Modulo Distribution (Default)

Instances dibagi secara round-robin:
- **Device 1**: Instances 0, 3, 6, 9, ...
- **Device 2**: Instances 1, 4, 7, 10, ...
- **Device 3**: Instances 2, 5, 8, 11, ...

**Keuntungan:**
- Workload balanced
- Mudah untuk tambah device

### 2. Type Distribution

Instances dibagi berdasarkan tipe:
- **Device 1**: Semua C instances (C1, C2)
- **Device 2**: Semua R instances (R1, R2)
- **Device 3**: Semua RC instances (RC1, RC2)

**Keuntungan:**
- Mudah track progress per tipe
- Cocok jika jumlah device = jumlah tipe

## 🔒 Thread Safety & Multi-Device Safety

- ✅ **Checkpoint file locking** - Aman untuk multiple devices
- ✅ **Atomic file operations** - Write ke temp file, lalu rename (atomic)
- ✅ **No race conditions** - Setiap device kerja pada instance berbeda
- ✅ **Instance-level separation** - Tidak ada konflik karena instance berbeda
- ✅ **Safe result files** - Setiap device menulis ke instance berbeda

## 📁 Shared Resources

### Checkpoint File
- **Location**: `results/checkpoint.json`
- **Shared**: Ya, semua device baca/tulis ke file yang sama
- **Safety**: File locking mencegah konflik

### Results Files
- **Location**: `results/experiments/{instance}/{instance}_results.json`
- **Shared**: Ya, tapi setiap device menulis instance berbeda
- **Safety**: Tidak ada konflik karena instance berbeda

### Device Config Files
- **Location**: `results/device_{device_id}_config.json`
- **Shared**: Tidak, setiap device punya file sendiri
- **Purpose**: Track assignment per device

## 🎯 Example: 3 Devices

### Setup:
```bash
python setup_multi_device.py --total-devices 3 --method modulo
```

### Distribution (56 instances total):
- **Device 1**: ~19 instances (C101, C104, C107, R101, R104, ...)
- **Device 2**: ~19 instances (C102, C105, C108, R102, R105, ...)
- **Device 3**: ~18 instances (C103, C106, C109, R103, R106, ...)

### Runtime:
- **Sequential (1 device)**: ~100 hours
- **3 devices parallel**: ~33 hours
- **Speedup**: ~3x

## ⚠️ Important Notes

1. **Shared Checkpoint**: 
   - **Best**: Gunakan network drive yang accessible semua device
   - **Alternative**: Sync checkpoint secara berkala (copy file)
   - **Note**: Checkpoint menggunakan atomic writes, aman untuk concurrent access
   - **Location**: `results/checkpoint.json` (sama di semua device)

2. **Device ID**: Harus unique dan konsisten
   - Gunakan nama yang jelas: "laptop", "server1", "desktop", dll.
   - Jangan ubah device_id setelah mulai (akan reassign)

3. **Total Devices**: Harus sama di semua device
   - Jika tambah device baru, update semua device config
   - Re-run setup script dengan total devices baru

4. **Checkpoint Sync**: 
   - Jika checkpoint di local, perlu sync manual
   - Atau gunakan shared network drive
   - Atau sync via git (tidak recommended untuk real-time)

## 🔍 Monitoring

### Check Device Assignment:
```bash
# Lihat assignment device ini
python -c "from src.evaluation.task_distributor import TaskDistributor; import yaml; config = yaml.safe_load(open('config.yaml')); d = TaskDistributor(config['device']['device_id'], config['device']['total_devices']); print(d.get_summary())"
```

### Check Overall Progress:
```bash
python check_progress.py
```

## 🛠️ Troubleshooting

### Problem: "Device assignment not found"
**Solution**: Run setup script atau biarkan auto-generate saat pertama run

### Problem: "Multiple devices working on same instance"
**Solution**: Pastikan device_id unique dan total_devices sama di semua device

### Problem: "Checkpoint file locked"
**Solution**: Normal, sistem akan retry. Pastikan file accessible dari semua device.

## 📝 Best Practices

1. **Use Network Drive**: Simpan checkpoint di network drive yang accessible semua device
2. **Unique Device IDs**: Gunakan nama yang jelas dan unique
3. **Monitor Progress**: Cek progress secara berkala di semua device
4. **Backup Checkpoint**: Backup checkpoint secara berkala
5. **Consistent Config**: Pastikan config.yaml sama di semua device (kecuali device_id)

## 🎉 Result

Setelah semua device selesai:
- Semua hasil akan ada di `results/experiments/`
- Summary akan aggregate semua hasil
- Tidak ada duplikasi atau konflik
- **True parallel execution!** 🚀


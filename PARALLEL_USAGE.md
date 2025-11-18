# Parallel Execution Guide

## 🚀 High Performance Computing Support

Sistem sekarang mendukung **parallel execution** untuk memanfaatkan semua core CPU yang tersedia!

## ✅ Fitur

- ✅ **Thread-safe checkpointing** - Tidak ada race condition
- ✅ **Automatic core detection** - Otomatis menggunakan semua core yang tersedia
- ✅ **Safe file I/O** - File locking untuk mencegah konflik
- ✅ **Resume support** - Tetap bisa resume meski parallel
- ✅ **Progress tracking** - Progress tetap ter-track dengan baik

## 📊 Konfigurasi

Edit `config.yaml`:

```yaml
evaluation:
  parallel: true  # Enable parallel execution
  n_jobs: null    # null = auto (CPU_COUNT - 1), atau set manual (e.g., 8)
  n_runs: 5       # Number of runs per method
```

### n_jobs Options:
- `null` atau `-1`: Auto (gunakan semua core - 1)
- `1`: Sequential (tidak parallel)
- `4`: Gunakan 4 cores
- `8`: Gunakan 8 cores
- dll.

## 🔒 Thread Safety

Sistem menggunakan:
1. **File locking** untuk checkpoint file
2. **Atomic file operations** (temp file + rename)
3. **Thread-safe checkpoint manager**
4. **Safe result collection**

**Tidak ada race condition!** ✅

## 📈 Performance

### Sequential (n_jobs=1):
- 1 run per waktu
- Cocok untuk debugging

### Parallel (n_jobs=auto):
- Multiple runs secara bersamaan
- **Speedup: ~N cores** (linear scaling)
- Contoh: 8 cores = ~8x faster untuk multiple runs

## 💡 Best Practices

### 1. Untuk HPC Cluster:
```yaml
evaluation:
  parallel: true
  n_jobs: 16  # Sesuaikan dengan jumlah cores yang dialokasikan
```

### 2. Untuk Laptop/Desktop:
```yaml
evaluation:
  parallel: true
  n_jobs: null  # Auto-detect (akan gunakan CPU_COUNT - 1)
```

### 3. Untuk Debugging:
```yaml
evaluation:
  parallel: false  # Atau n_jobs: 1
```

## ⚠️ Catatan Penting

1. **Memory Usage**: Parallel execution menggunakan lebih banyak RAM
   - Setiap worker membutuhkan memory untuk GA instance
   - Monitor memory usage jika banyak cores

2. **Checkpoint Safety**: 
   - Checkpoint file menggunakan file locking
   - Aman untuk multiple processes
   - Tidak akan corrupt meski parallel

3. **Convergence History**:
   - Saat ini, convergence history CSV tidak disimpan dalam parallel mode
   - Hanya metrics summary yang disimpan
   - Untuk convergence history, gunakan sequential mode

4. **Optimal n_jobs**:
   - Jangan set lebih dari jumlah physical cores
   - Leave 1-2 cores free untuk OS
   - Contoh: 8-core CPU → n_jobs: 6-7

## 🔍 Monitoring

Parallel execution tetap bisa di-monitor:
```bash
# Cek progress
python check_progress.py

# Monitor real-time
python monitor_experiment.py
```

## 🛠️ Troubleshooting

### Problem: "Too many open files"
**Solution**: Kurangi `n_jobs` atau increase system file limit

### Problem: "Out of memory"
**Solution**: Kurangi `n_jobs` atau increase system RAM

### Problem: "Checkpoint file locked"
**Solution**: Normal behavior, sistem akan retry. Jika stuck, restart.

## 📝 Example

```python
# Sequential (old way)
evaluator = ExperimentEvaluator(instance, n_runs=5, parallel=False)

# Parallel (new way)
evaluator = ExperimentEvaluator(instance, n_runs=5, parallel=True, n_jobs=8)
```

## 🎯 Performance Comparison

| Mode | n_runs=5 | n_runs=10 | Speedup |
|------|----------|-----------|---------|
| Sequential | 5 min | 10 min | 1x |
| Parallel (4 cores) | ~1.25 min | ~2.5 min | ~4x |
| Parallel (8 cores) | ~0.6 min | ~1.25 min | ~8x |

*Estimasi berdasarkan GA runtime ~1 min per run*


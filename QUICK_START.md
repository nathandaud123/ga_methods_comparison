# Quick Start - Jalankan Eksperimen

## ✅ Cara Paling Sederhana

### Windows (PowerShell atau CMD):
```bash
python main.py --config config.yaml
```

### Atau gunakan batch file:
```bash
quick_start.bat
```

### Atau gunakan PowerShell script:
```powershell
.\start_experiment.ps1
```

## 🔍 Cara Cek Apakah Berjalan

### 1. Lihat Output di Terminal
Jika eksperimen berjalan, Anda akan melihat output seperti:
```
================================================================================
Genetic Algorithm Method Comparison Study
================================================================================

Processing instance: C101.csv
...
```

### 2. Cek Status
```bash
# Quick status check
python status_check.py

# Detailed progress
python check_progress.py
```

### 3. Monitor Real-time
```bash
# Auto-refresh setiap 5 detik
python monitor_experiment.py
```

### 4. Manual Check
```bash
# Windows PowerShell - Cek checkpoint update time
Get-Item results\checkpoint.json | Select-Object LastWriteTime

# Cek log file terbaru
Get-ChildItem results\logs\experiment_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

## ⚠️ Jika Tidak Ada Output

1. **Cek apakah ada error**: Lihat terminal untuk pesan error
2. **Cek Python**: `python --version`
3. **Cek dependencies**: `pip install -r requirements.txt`
4. **Cek config**: Pastikan `config.yaml` ada dan valid
5. **Cek status**: `python status_check.py` untuk diagnosis

## 📊 Monitoring Tools

| Script | Purpose |
|--------|---------|
| `status_check.py` | Quick status check (checkpoint age, log status) |
| `check_progress.py` | Detailed progress summary |
| `monitor_experiment.py` | Real-time monitoring (auto-refresh) |

## 🛑 Stop Eksperimen

Tekan `Ctrl+C` di terminal yang menjalankan eksperimen.
Checkpoint akan otomatis tersimpan, jadi bisa resume nanti.

## 🔄 Resume Eksperimen

Jalankan lagi dengan perintah yang sama:
```bash
python main.py --config config.yaml
```

Sistem akan otomatis:
- ✅ Skip instance yang sudah selesai
- ✅ Skip method yang sudah selesai  
- ✅ Resume dari run terakhir yang belum selesai

## ⚡ Parallel Execution

Untuk menggunakan semua CPU cores (HPC):
```yaml
# config.yaml
evaluation:
  parallel: true
  n_jobs: null  # Auto-detect, atau set manual (e.g., 8)
```

Lihat `PARALLEL_USAGE.md` untuk detail lengkap.


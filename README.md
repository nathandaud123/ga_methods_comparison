# Genetic Algorithm Method Comparison Study (Simplified)

Sistem perbandingan operator Genetic Algorithm untuk Vehicle Routing Problem menggunakan dataset benchmark Solomon.

## 🎯 Fitur Utama

1. **Semua Kombinasi**: Semua kombinasi dari representation, selection, crossover, dan mutation
2. **5 Runs per Kombinasi**: Setiap kombinasi dijalankan 5 kali untuk statistik yang reliable
3. **Simpan Setiap Generasi**: History fitness dan diversity disimpan untuk setiap run
4. **Rata-rata per Generasi**: Rata-rata dari 5 runs dihitung dan disimpan per generasi
5. **Semua Instance**: Semua kombinasi diuji pada semua instance Solomon

## 📋 Skenario Eksperimen

1. **Generate Kombinasi**: Semua kombinasi representation × selection × crossover × mutation
2. **Run 5x**: Setiap kombinasi dijalankan 5 kali (independent runs)
3. **Simpan History**: Setiap generasi dari setiap run disimpan
4. **Hitung Rata-rata**: Rata-rata per generasi dari 5 runs dihitung
5. **Simpan Hasil**: Hasil disimpan dalam format CSV dan JSON

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run dengan semua datasets (sequential)
python main.py --config config.yaml

# Run dengan parallel execution (4 workers)
# Edit config.yaml: n_jobs: 4
python main.py --config config.yaml

# Run dengan auto-detect CPU cores
# Edit config.yaml: n_jobs: 0
python main.py --config config.yaml

# Run instance tertentu saja
python main.py --config config.yaml --instance C101
python main.py --config config.yaml --instance C102
```

## 🔀 Multi-Terminal / Multi-Instance Execution

Untuk menjalankan beberapa instance secara paralel di terminal berbeda (misalnya di AI Center):

### 1. Run Instance Spesifik

Setiap terminal menjalankan instance yang berbeda:

```bash
# Terminal 1
python main.py --config config.yaml --instance C101

# Terminal 2
python main.py --config config.yaml --instance C102

# Terminal 3
python main.py --config config.yaml --instance C103
```

**Catatan:**
- Setiap instance akan menggunakan checkpoint file terpisah: `checkpoint_<instance>.json`
- Hasil tetap disimpan di `results/<instance>/<instance>_results.json`
- Tidak ada konflik karena setiap instance menggunakan checkpoint file sendiri

### 2. Merge Checkpoints

Setelah semua instance selesai, merge checkpoint files menjadi satu:

```bash
# Merge semua checkpoint_*.json menjadi checkpoint.json
python merge_checkpoints.py --results-dir results

# Atau dengan output file custom
python merge_checkpoints.py --results-dir results --output results/checkpoint_merged.json
```

**Fungsi merge:**
- Menggabungkan semua `checkpoint_<instance>.json` menjadi satu `checkpoint.json`
- Menggabungkan `completed_methods` dari semua instance
- Menggabungkan `completed_instances` dari semua instance
- Menampilkan summary per instance

## ✅ Checkpoint & Resume

Sistem mendukung checkpoint/resume:
- **Auto-save**: Setelah setiap kombinasi selesai
- **Auto-resume**: Jika terminate, jalankan lagi dan akan lanjut dari yang belum selesai
- **Skip completed**: Kombinasi yang sudah selesai akan di-skip otomatis

**Checkpoint files:**
- **All instances mode**: `results/checkpoint.json`
- **Single instance mode** (`--instance`): `results/checkpoint_<instance>.json`
  - Contoh: `checkpoint_C101.json`, `checkpoint_C102.json`
  - Setiap instance memiliki checkpoint file sendiri (tidak numpuk)

## ⚡ Parallel Execution

Sistem mendukung parallel execution untuk mempercepat proses:

**Konfigurasi di `config.yaml`:**
```yaml
evaluation:
  n_jobs: null  # Sequential (default)
  # n_jobs: 1   # Sequential
  # n_jobs: 0   # Auto: use all CPU cores - 1
  # n_jobs: 4   # Use 4 parallel workers
  # n_jobs: 8   # Use 8 parallel workers
```

**Cara kerja:**
- `n_jobs: null` atau `1`: Sequential execution (1 kombinasi per waktu)
- `n_jobs: 0`: Auto-detect CPU cores (gunakan semua core - 1)
- `n_jobs: N` (N > 1): Gunakan N parallel workers

**Contoh:**
- 5 runs per kombinasi dengan `n_jobs: 5` → 5 runs dijalankan parallel
- Speedup: ~Nx faster (dengan overhead minimal)

## 📁 Struktur Output

```
results/
├── {instance_name}/
│   ├── {method_name}_convergence.csv  # History per run + average
│   └── {instance_name}_results.json   # Summary results
```

### Format CSV Convergence

Setiap file `{method_name}_convergence.csv` berisi:
- `generation`: Nomor generasi (1, 2, 3, ...)
- `fitness_run_1` sampai `fitness_run_5`: Fitness per run per generasi
- `diversity_run_1` sampai `diversity_run_5`: Diversity per run per generasi
- `fitness_average`: Rata-rata fitness per generasi (dari 5 runs)
- `diversity_average`: Rata-rata diversity per generasi (dari 5 runs)

### Format JSON Results

Setiap file `{instance_name}_results.json` berisi:
```json
{
  "method_name": {
    "method_name": "...",
    "average_fitness_history": [...],  // Rata-rata per generasi
    "average_diversity_history": [...],
    "best_fitness": 1234.56,
    "mean_fitness": 1234.56,
    "std_fitness": 12.34,
    "runtime": 123.45,
    "n_runs": 5
  }
}
```

## ⚙️ Konfigurasi

Edit `config.yaml` untuk mengubah:
- GA parameters (population_size, max_generations, dll)
- Representations, selection methods, crossover methods, mutation methods
- Number of runs per combination (default: 5)

## 📊 Contoh Output

```
================================================================================
Processing instance: C101
================================================================================
Instance: C101
Type: C
Customers: 100
Capacity: 200.0

============================================================
Evaluating: permutation_tournament_pmx_swap
============================================================
  Run 1/5... Best: 1234.56
  Run 2/5... Best: 1235.12
  Run 3/5... Best: 1234.89
  Run 4/5... Best: 1235.45
  Run 5/5... Best: 1234.23
  Saved convergence history to results/C101/permutation_tournament_pmx_swap_convergence.csv
  Average final fitness: 1234.85 ± 0.45
  Best fitness: 1234.23
  Runtime: 45.67s
```

## 🔍 Analisis Data

Data yang disimpan dapat dianalisis untuk:
- Convergence behavior per kombinasi
- Perbandingan performa antar kombinasi
- Statistical significance testing
- Visualization convergence curves

## 📝 Catatan

- Sistem ini lebih sederhana dari versi sebelumnya
- Tidak ada checkpoint/resume (untuk kesederhanaan)
- Tidak ada parallel execution (dapat ditambahkan jika perlu)
- Fokus pada data collection yang lengkap untuk analisis


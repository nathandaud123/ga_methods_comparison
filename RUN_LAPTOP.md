# Cara Run di Laptop

## Opsi 1: Run Standalone (Tanpa Multi-Device)

Ini adalah cara paling sederhana untuk run di laptop sendiri.

### Langkah-langkah:

1. **Pastikan config.yaml sudah benar:**
   ```yaml
   device:
     device_id: null  # null = standalone mode
     total_devices: 1
   
   evaluation:
     parallel: true
     n_jobs: null     # Auto-detect (akan pakai CPU_COUNT - 1)
     parallel_methods: false  # Sequential methods, parallel runs per method
   ```

2. **Jalankan:**
   ```bash
   python main.py --config config.yaml
   ```

3. **Atau dengan instance tertentu:**
   Edit `config.yaml`:
   ```yaml
   dataset:
     instances:
       - "C101"
       - "C102"
   ```
   Lalu run:
   ```bash
   python main.py --config config.yaml
   ```

## Opsi 2: Run dengan Multi-Device Mode (Laptop + AI Center)

Jika ingin split workload dengan AI Center:

### Langkah-langkah:

1. **Edit config.yaml:**
   ```yaml
   device:
     device_id: "laptop"  # Set ID untuk laptop
     total_devices: 2      # Total: Laptop + AI Center
   ```

2. **Di AI Center, edit config juga:**
   ```yaml
   device:
     device_id: "aicenter"
     total_devices: 2
   ```

3. **Jalankan di laptop:**
   ```bash
   python main.py --config config.yaml
   ```

   Instances akan otomatis di-split:
   - Laptop: instances 0, 2, 4, 6, ... (even indices)
   - AI Center: instances 1, 3, 5, 7, ... (odd indices)

## Opsi 3: Run dengan Optuna Tuning

Jika ingin test tuning flow:

1. **Edit config.yaml:**
   ```yaml
   optuna:
     enabled: true
     n_trials: 10  # Kecilkan untuk test cepat
     timeout: 600  # 10 menit
   ```

2. **Atau gunakan test config:**
   ```bash
   python main.py --config config_test_tuning.yaml
   ```

## Tips untuk Laptop:

1. **Monitor Progress:**
   ```bash
   python check_progress.py
   ```

2. **View Results:**
   ```bash
   python view_results.py C101
   ```

3. **Check Checkpoint:**
   File: `results/checkpoint.json`
   - Lihat progress semua instances
   - Lihat completed methods

4. **Resume jika terhenti:**
   - Program otomatis resume dari checkpoint
   - Tidak perlu restart dari awal

## Performance Settings untuk Laptop:

- **n_jobs: null** → Auto-detect (biasanya CPU_COUNT - 1)
- **parallel_methods: false** → Sequential methods, parallel runs
- **n_runs: 5** → 5 runs per method (bisa dikurangi untuk test)

## Output Files:

Setelah run, hasil akan ada di:
- `results/experiments/{instance}/{instance}_results.json` - Results
- `results/plots/{instance}/` - Visualizations
- `results/convergence/{instance}/` - Convergence history CSV
- `results/tuning/{instance}/` - Optuna tuning results (jika enabled)
- `results/routes/{instance}/` - Best route visualizations


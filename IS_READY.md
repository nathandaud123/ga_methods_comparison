# Apakah Siap untuk Run?

## ✅ Ya, Siap!

Perintah `python main.py --config config.yaml` **AKAN menjalankan eksperimen**.

## 📋 Yang Akan Terjadi

1. **Load config** dari `config.yaml`
2. **Auto-discover datasets** dari `data/solomon/` (jika instances: [])
3. **Filter instances** jika multi-device mode (device_id set)
4. **Run GA experiments** untuk setiap instance
5. **Save results** ke `results/experiments/`
6. **Save checkpoint** setelah setiap run
7. **Generate visualizations** (plots, routes)

## ⚙️ Mode yang Akan Aktif

### Jika device_id = null (default):
- **Single device mode**
- Process semua instances
- Parallel execution jika `evaluation.parallel: true`

### Jika device_id set (e.g., "laptop", "aicenter"):
- **Multi-device mode**
- Hanya process instances yang di-assign ke device ini
- Tidak ada overlap dengan device lain

## 🚀 Quick Start

### Untuk Single Device (Laptop saja):
```bash
# Config sudah OK (device_id: null)
python main.py --config config.yaml
```

### Untuk Multi-Device (Laptop + AI Center):

**Laptop:**
```yaml
# Edit config.yaml
device:
  device_id: "laptop"
  total_devices: 2
```
```bash
python main.py --config config.yaml
```

**AI Center:**
```yaml
# Edit config.yaml
device:
  device_id: "aicenter"
  total_devices: 2

evaluation:
  parallel: true
  n_jobs: 250  # 256 cores - 6 untuk OS
```
```bash
python main.py --config config.yaml
```

## ⚠️ Sebelum Run

1. **Cek config**: Pastikan `config.yaml` sesuai kebutuhan
2. **Cek datasets**: Pastikan file CSV ada di `data/solomon/`
3. **Cek dependencies**: `pip install -r requirements.txt`
4. **Cek checkpoint**: Jika ada, akan auto-resume

## ✅ Ready!

**Ya, langsung run aja!**
```bash
python main.py --config config.yaml
```

Eksperimen akan mulai dan bisa dihentikan kapan saja (Ctrl+C), checkpoint akan tersimpan.


# Quick Multi-Device Setup

## 🚀 Setup Cepat untuk 3 Devices

### 1. Setup (Sekali Saja)

```bash
# Di device utama (atau device manapun)
python setup_multi_device.py --total-devices 3 --method modulo
```

### 2. Clone di Setiap Device

```bash
# Device 1, 2, 3 - sama semua
git clone <your-repo>
cd ga_method_comparison
```

### 3. Edit config.yaml di Setiap Device

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

### 4. Run di Setiap Device

```bash
# Sama di semua device
python main.py --config config.yaml
```

## ✅ Hasil

- **Device 1**: Kerja pada ~19 instances (C101, C104, R101, R104, ...)
- **Device 2**: Kerja pada ~19 instances (C102, C105, R102, R105, ...)
- **Device 3**: Kerja pada ~18 instances (C103, C106, R103, R106, ...)

**Tidak ada konflik!** Setiap device kerja pada instance berbeda.

## 📊 Progress Tracking

Semua device menulis ke checkpoint yang sama:
- Progress ter-track di semua device
- Tidak ada duplikasi
- Bisa cek progress dari device manapun: `python check_progress.py`

## ⚠️ Important

1. **Checkpoint Location**: Pastikan semua device bisa akses `results/checkpoint.json`
   - Network drive: Best option
   - Local: Copy checkpoint secara berkala

2. **Device ID**: Harus unique dan konsisten
   - Jangan ubah setelah mulai run

3. **Total Devices**: Harus sama di semua device

## 🎉 Done!

Setelah semua selesai, hasil akan ada di `results/experiments/` dan bisa di-aggregate!


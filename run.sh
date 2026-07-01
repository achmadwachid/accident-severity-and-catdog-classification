#!/bin/bash

# Nama folder virtual environment yang akan digunakan
VENV_DIR=".venv"

echo "=== Memulai Runner Aplikasi Streamlit ==="

# 1. Cari interpreter Python yang kompatibel (>= 3.10)
PYTHON_EXE=""
for py_cmd in python3.12 python3.11 python3.10; do
    if command -v "$py_cmd" &> /dev/null; then
        PYTHON_EXE="$py_cmd"
        break
    fi
done

# Jika tidak ditemukan yang >= 3.10, default ke python3 biasa
if [ -z "$PYTHON_EXE" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_EXE="python3"
    else
        echo -e "\033[0;31m[ERROR] Python tidak ditemukan di sistem Anda.\033[0m"
        exit 1
    fi
fi

# 2. Cek apakah folder virtual environment sudah ada dan versinya cocok
if [ -d "$VENV_DIR" ]; then
    VENV_PYTHON_VER=$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    MAJOR=$(echo "$VENV_PYTHON_VER" | cut -d. -f1)
    MINOR=$(echo "$VENV_PYTHON_VER" | cut -d. -f2)
    
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
        echo -e "\033[0;32m[INFO] Virtual environment '$VENV_DIR' (Python $VENV_PYTHON_VER) ditemukan.\033[0m"
    else
        echo -e "\033[0;33m[WARNING] Virtual environment '$VENV_DIR' menggunakan Python $VENV_PYTHON_VER (kurang dari 3.10).\033[0m"
        echo -e "[INFO] Menghapus virtual environment lama untuk diperbarui dengan Python yang kompatibel..."
        rm -rf "$VENV_DIR"
    fi
fi

# Cek alternatif nama 'venv' jika .venv tidak ada
if [ ! -d "$VENV_DIR" ] && [ -d "venv" ]; then
    VENV_DIR="venv"
    VENV_PYTHON_VER=$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    MAJOR=$(echo "$VENV_PYTHON_VER" | cut -d. -f1)
    MINOR=$(echo "$VENV_PYTHON_VER" | cut -d. -f2)
    
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
        echo -e "\033[0;32m[INFO] Virtual environment 'venv' (Python $VENV_PYTHON_VER) ditemukan.\033[0m"
    else
        echo -e "\033[0;33m[WARNING] Virtual environment 'venv' menggunakan Python $VENV_PYTHON_VER (kurang dari 3.10).\033[0m"
        echo -e "[INFO] Menghapus virtual environment lama..."
        rm -rf "venv"
        VENV_DIR=".venv"
    fi
fi

# 3. Membuat virtual environment baru jika tidak ada
if [ ! -d "$VENV_DIR" ]; then
    echo -e "[INFO] Menggunakan interpreter: $PYTHON_EXE ($("$PYTHON_EXE" --version))"
    echo -e "\033[0;33m[INFO] Membuat virtual environment baru ($VENV_DIR)...\033[0m"
    "$PYTHON_EXE" -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31m[ERROR] Gagal membuat virtual environment.\033[0m"
        exit 1
    fi
    echo -e "\033[0;32m[SUCCESS] Virtual environment '$VENV_DIR' berhasil dibuat.\033[0m"
    INSTALL_DEPS=true
fi

# 4. Mengaktifkan virtual environment
echo "[INFO] Mengaktifkan virtual environment..."
source "$VENV_DIR/bin/activate"

# 5. Cek/Install dependensi
if [ "$INSTALL_DEPS" = true ] || ! python3 -c "import streamlit; import xgboost; import joblib; import sklearn" &> /dev/null; then
    echo -e "\033[0;33m[INFO] Menginstall/memperbarui dependensi dari requirements.txt...\033[0m"
    pip install --upgrade pip
    
    # Tambahkan joblib ke requirements jika belum terdaftar
    if ! grep -q "joblib" requirements.txt; then
        echo "joblib==1.4.2" >> requirements.txt
    fi
    # Tambahkan xgboost ke requirements jika belum terdaftar
    if ! grep -q "xgboost" requirements.txt; then
        echo "xgboost==2.0.3" >> requirements.txt
    fi
    # Tambahkan scikit-learn ke requirements jika belum terdaftar
    if ! grep -q "scikit-learn" requirements.txt; then
        echo "scikit-learn" >> requirements.txt
    fi
    
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31m[ERROR] Gagal menginstall dependensi.\033[0m"
        exit 1
    fi
    echo -e "\033[0;32m[SUCCESS] Dependensi berhasil diinstall.\033[0m"
fi

# 6. Menjalankan aplikasi streamlit
echo -e "\033[0;34m[INFO] Menjalankan aplikasi Streamlit...\033[0m"
streamlit run app.py

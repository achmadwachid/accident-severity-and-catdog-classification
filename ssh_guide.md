# Panduan Multi-Akun GitHub (SSH Key)

Berikut adalah panduan penggunaan dua akun GitHub yang telah dikonfigurasi pada komputer ini.

---

## 1. Konfigurasi SSH yang Tersimpan (`~/.ssh/config`)

```text
# Akun Pertama (Utama - ahmazroot)
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 5

# Akun Kedua (achmadwachid / akhmasroot)
Host github-akhmasroot
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_akhmasroot
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

---

## 2. Cara Menggunakan (Clone & Remote)

### A. Menggunakan Akun Pertama (`ahmazroot`)
*   **Clone Repository Baru:**
    ```bash
    git clone git@github.com:USERNAME_AKUN_1/nama-repo.git
    ```
*   **Mengubah Project yang Sudah Ada Agar Menggunakan SSH Akun Pertama:**
    ```bash
    git remote set-url origin git@github.com:USERNAME_AKUN_1/nama-repo.git
    ```

### B. Menggunakan Akun Kedua (`achmadwachid / akhmasroot`)
*   **Clone Repository Baru:**
    Ganti bagian `@github.com` menjadi `@github-akhmasroot`:
    ```bash
    git clone git@github-akhmasroot:USERNAME_AKUN_2/nama-repo.git
    ```
*   **Mengubah Project yang Sudah Ada Agar Menggunakan SSH Akun Kedua:**
    ```bash
    git remote set-url origin git@github-akhmasroot:USERNAME_AKUN_2/nama-repo.git
    ```
*   **Penting! Atur Nama & Email Lokal di Folder Project Anda:**
    Jalankan perintah ini di dalam folder repository akun kedua agar nama pembuat commit benar:
    ```bash
    git config user.name "achmadwachid"
    git config user.email "rootakhmas@gmail.com"
    ```

---

## 3. Tes Koneksi SSH
Jalankan perintah ini untuk memastikan SSH key Anda terhubung ke akun GitHub masing-masing dengan sukses:

*   **Tes Akun Pertama:**
    ```bash
    ssh -T git@github.com
    ```
*   **Tes Akun Kedua:**
    ```bash
    ssh -T git@github-akhmasroot
    ```

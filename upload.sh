#!/bin/bash

# ===========================================
# SCRIPT UPLOAD FILE KE PENYIMPANAN PUBLIK
# ===========================================

# 1. Upload ke file.io (file otomatis terhapus setelah 1x download)
upload_fileio() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        echo "File tidak ditemukan: $file_path"
        return 1
    fi
    
    echo "Uploading ke file.io..."
    response=$(curl -s -F "file=@$file_path" https://file.io)
    echo "Response: $response"
    
    # Extract link dari response JSON
    link=$(echo "$response" | grep -o '"link":"[^"]*' | cut -d'"' -f4)
    if [ ! -z "$link" ]; then
        echo "✅ Upload berhasil!"
        echo "🔗 Link: $link"
        echo "⚠️  File akan terhapus setelah 1x download"
    else
        echo "❌ Upload gagal"
    fi
}

# 2. Upload ke 0x0.st (file storage sementara)
upload_0x0() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        echo "File tidak ditemukan: $file_path"
        return 1
    fi
    
    echo "Uploading ke 0x0.st..."
    response=$(curl -s -F "file=@$file_path" https://0x0.st)
    
    if [[ $response == http* ]]; then
        echo "✅ Upload berhasil!"
        echo "🔗 Link: $response"
        echo "⚠️  File disimpan sementara"
    else
        echo "❌ Upload gagal: $response"
    fi
}

# 3. Upload ke tmpfiles.org
upload_tmpfiles() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        echo "File tidak ditemukan: $file_path"
        return 1
    fi
    
    echo "Uploading ke tmpfiles.org..."
    response=$(curl -s -F "file=@$file_path" https://tmpfiles.org/api/v1/upload)
    echo "Response: $response"
    
    # Extract URL dari response
    url=$(echo "$response" | grep -o '"url":"[^"]*' | cut -d'"' -f4)
    if [ ! -z "$url" ]; then
        echo "✅ Upload berhasil!"
        echo "🔗 Link: $url"
    else
        echo "❌ Upload gagal"
    fi
}

# 4. Upload ke uguu.se
upload_uguu() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        echo "File tidak ditemukan: $file_path"
        return 1
    fi
    
    echo "Uploading ke uguu.se..."
    response=$(curl -s -F "files[]=@$file_path" https://uguu.se/upload.php)
    echo "Response: $response"
    
    # Extract URL dari response JSON
    url=$(echo "$response" | grep -o '"url":"[^"]*' | cut -d'"' -f4)
    if [ ! -z "$url" ]; then
        echo "✅ Upload berhasil!"
        echo "🔗 Link: $url"
    else
        echo "❌ Upload gagal"
    fi
}

# 5. Upload ke transfer.sh
upload_transfer() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        echo "File tidak ditemukan: $file_path"
        return 1
    fi
    
    filename=$(basename "$file_path")
    echo "Uploading ke transfer.sh..."
    response=$(curl -s --upload-file "$file_path" "https://transfer.sh/$filename")
    
    if [[ $response == https* ]]; then
        echo "✅ Upload berhasil!"
        echo "🔗 Link: $response"
        echo "⚠️  File akan tersedia selama 14 hari"
    else
        echo "❌ Upload gagal: $response"
    fi
}

# Fungsi utama dengan menu
main() {
    if [ $# -eq 0 ]; then
        echo "📁 UPLOAD FILE KE PENYIMPANAN PUBLIK"
        echo "======================================"
        echo "Usage: $0 <file_path> [service]"
        echo ""
        echo "Services tersedia:"
        echo "1. fileio    - file.io (1x download)"
        echo "2. 0x0       - 0x0.st (temporary)"
        echo "3. tmpfiles  - tmpfiles.org"
        echo "4. uguu      - uguu.se"
        echo "5. transfer  - transfer.sh (14 hari)"
        echo ""
        echo "Contoh:"
        echo "$0 document.pdf fileio"
        echo "$0 image.jpg transfer"
        return 1
    fi
    
    file_path="$1"
    service="${2:-fileio}"
    
    echo "📤 Mengupload file: $file_path"
    echo "🌐 Service: $service"
    echo ""
    
    case "$service" in
        "fileio"|"1")
            upload_fileio "$file_path"
            ;;
        "0x0"|"2")
            upload_0x0 "$file_path"
            ;;
        "tmpfiles"|"3")
            upload_tmpfiles "$file_path"
            ;;
        "uguu"|"4")
            upload_uguu "$file_path"
            ;;
        "transfer"|"5")
            upload_transfer "$file_path"
            ;;
        *)
            echo "❌ Service tidak dikenal: $service"
            echo "Gunakan: fileio, 0x0, tmpfiles, uguu, atau transfer"
            return 1
            ;;
    esac
}

# Jalankan script
main "$@"

# ===========================================
# CONTOH PENGGUNAAN MANUAL:
# ===========================================

# Upload ke file.io
# curl -F "file=@dokumen.pdf" https://file.io

# Upload ke 0x0.st  
# curl -F "file=@gambar.jpg" https://0x0.st

# Upload ke transfer.sh
# curl --upload-file "video.mp4" https://transfer.sh/video.mp4

# Upload ke uguu.se
# curl -F "files[]=@musik.mp3" https://uguu.se/upload.php

# Upload ke tmpfiles.org
# curl -F "file=@arsip.zip" https://tmpfiles.org/api/v1/upload
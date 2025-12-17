# 🔧 Git Repository Cleanup Guide

## ⚠️ **Vấn đề hiện tại:**

1. **File `yolov8m-pose.pt` (51MB)** đã được push lên GitHub
2. **Thư mục `.venv/`** đã được add vào staging
3. Chưa có `.gitignore`

---

## ✅ **Giải pháp - 3 BƯỚC:**

### **BƯỚC 1: Unstage files không cần thiết**

```bash
# Remove .venv từ staging (nhưng giữ file local)
git reset HEAD .venv/

# Check status
git status
```

---

### **BƯỚC 2: Commit .gitignore và code mới**

```bash
# Add gitignore
git add .gitignore

# Add source code mới
git add src/ PROJECT_STRUCTURE.md README.md

# Commit
git commit -m "Refactor: Add minimal geometric analysis and update structure

- Add geometry/ module with angle calculation
- Add evaluation/ module with pose scoring
- Update project structure to hybrid ML + geometric approach
- Add comprehensive .gitignore"
```

---

### **BƯỚC 3: Remove yolov8m-pose.pt từ Git history**

**⚠️ LƯU Ý:** File này ĐÃ được push lên GitHub rồi, cần xóa khỏi history.

#### **Option A: Đơn giản - Xóa và hướng dẫn tải lại**

```bash
# 1. Remove file khỏi git (nhưng giữ local)
git rm --cached yolov8m-pose.pt

# 2. Commit
git commit -m "Remove large model file from git tracking"

# 3. Push
git push origin main
```

**Sau đó thêm vào README:**
```markdown
## Tải Model Weights

Do giới hạn GitHub (file >50MB), bạn cần tải YOLO model:

```bash
# Tự động tải (recommended)
python -c "from ultralytics import YOLO; YOLO('yolov8m-pose.pt')"

# Hoặc tải thủ công
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt
```
```

#### **Option B: Advanced - Xóa hoàn toàn khỏi history (nếu cần)**

**⚠️ NGUY HIỂM:** Chỉ làm nếu repo chưa có người khác clone!

```bash
# Sử dụng git filter-branch (không khuyến khích)
# Hoặc BFG Repo-Cleaner
java -jar bfg.jar --delete-files yolov8m-pose.pt
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```

---

## 📋 **RECOMMENDED ACTION:**

### **Làm theo Option A (Đơn giản & An toàn):**

```bash
# 1. Unstage .venv
git reset HEAD .venv/

# 2. Add gitignore và source mới
git add .gitignore src/ PROJECT_STRUCTURE.md README.md

# 3. Commit
git commit -m "Refactor: Add minimal geometric analysis

- Add geometry/ and evaluation/ modules
- Update hybrid ML + geometric architecture
- Add .gitignore for model files"

# 4. Remove model file từ tracking
git rm --cached yolov8m-pose.pt

# 5. Commit removal
git commit -m "Remove large model file from git

Model weights should be downloaded separately.
See README for download instructions."

# 6. Push
git push origin main
```

---

## 📝 **Cập nhật README.md**

Thêm section hướng dẫn download model:

```markdown
## 🔧 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download YOLO model
```bash
# Auto-download (recommended)
python -c "from ultralytics import YOLO; YOLO('yolov8m-pose.pt')"
```

Model sẽ được tự động tải về (~51MB).
```

---

## ✅ **Kết quả:**

- ✅ `.gitignore` đã tạo
- ✅ `.venv/` không được track
- ✅ `*.pt` files không được track
- ✅ Code mới được commit
- ✅ Model file user tự download

**File size trên GitHub: Giảm từ ~51MB → <1MB** 🎉

# Booking Agent

Project thực hành agent đặt vé máy bay: ReAct, Plan-then-Execute, Hybrid,
**Challenge A: persistent memory** qua nhiều session và **Challenge B: context management**
(compact session dài, Code Mode trên nhiều file log).

Model thật được gọi qua OpenAI hoặc endpoint tương thích như 9router. Chuyến bay,
giữ chỗ và thanh toán là mô phỏng. Dữ liệu bài tập có tuyến SGN → DAD ngày cố định
07/10/2026.

## Cài đặt

Chạy PowerShell trong thư mục project, dùng Python 3.10 trở lên:

```powershell
python -m venv demo/.venv
.\demo\.venv\Scripts\python.exe -m pip install -r demo/requirements-ui.txt
if (-not (Test-Path -LiteralPath demo/.env)) { Copy-Item demo/.env.example demo/.env }
```

Điền `OPENAI_API_KEY`, `OPENAI_BASE_URL` và `FLIGHT_MODEL` trong `demo/.env`.
Với 9router, lấy API key, endpoint và model ID từ dashboard của instance đang dùng.

## Chạy chatbot

```powershell
.\demo\.venv\Scripts\python.exe -m streamlit run demo/chat_app.py --server.address 127.0.0.1 --server.port 8501
```

Mở `http://127.0.0.1:8501`.

## Chạy Challenge A

```powershell
.\demo\.venv\Scripts\python.exe -m streamlit run 'Challenge A/app.py' --server.address 127.0.0.1 --server.port 8502
```

Mở `http://127.0.0.1:8502`. Bản này dùng chung cấu hình trong `demo/.env`.
Memory JSON lưu quy ước đang áp dụng và lịch sử thay thế; tab **Bộ nhớ & bằng chứng**
hiển thị memory thực sự được nạp vào context.

## Chạy Challenge B

```powershell
.\demo\.venv\Scripts\python.exe -m streamlit run 'Challenge B/app.py' --server.address 127.0.0.1 --server.port 8503
```

Mở `http://127.0.0.1:8503`. Agent điều tra log giả của hệ thống đặt vé, sinh tự động vào
`Challenge B/data/`. Chọn chế độ `native` (đọc từng file) hoặc `code` (chạy code, log thô không
vào context); tab **Context & token** hiển thị input tokens từng lần gọi, compact và checkpoint.

Xem [hướng dẫn Challenge A](<Challenge A/README.md>), [hướng dẫn Challenge B](<Challenge B/README.md>)
và [hướng dẫn agent gốc](demo/README.md).

## Kiểm tra

```powershell
.\demo\.venv\Scripts\python.exe -m unittest discover -s demo -p 'test_*.py'
.\demo\.venv\Scripts\python.exe -m unittest discover -s 'Challenge A' -p 'test_*.py'
.\demo\.venv\Scripts\python.exe -m unittest discover -s 'Challenge B' -p 'test_*.py'
```

Kiểm chứng hai yêu cầu Challenge A bằng API thật, với memory kiểm thử riêng:

```powershell
.\demo\.venv\Scripts\python.exe 'Challenge A/verify_live.py' --live
```

Kiểm chứng B1 và B2 của Challenge B bằng API thật (tốn token; mặc định 10 file log cho B2):

```powershell
.\demo\.venv\Scripts\python.exe 'Challenge B/verify_live.py' --live
```

## Cấu trúc

- `demo/`: agent, tools, harness, chatbot và bộ kiểm tra.
- `Challenge A/`: persistent memory, giao diện và kiểm chứng riêng.
- `Challenge B/`: compact có checkpoint, Code Mode `run_code`, log giả có đáp án, giao diện và kiểm chứng riêng.
- `Demo_gốc/`: các ví dụ agent ban đầu.

`.gitignore` loại trừ `docs`, `img`, `output`, `reports`, `tmp`, file `.env`,
virtualenv, cache, dữ liệu memory cá nhân và log sinh ra của Challenge B. File `.env.example` được giữ để cấu hình
trên máy mới. Báo cáo kiểm chứng được sinh lại bằng `verify_live.py`.

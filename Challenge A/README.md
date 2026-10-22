# Challenge A: persistent memory

Folder này mở rộng booking agent hiện có để chứng minh hai yêu cầu:

1. Cuộc chat mới vẫn áp dụng quy ước dự án đã thống nhất.
2. Sau khi quy ước thay đổi, cuộc chat tiếp theo dùng quy ước mới.

## Chạy giao diện

Từ thư mục gốc `booking_agent`, dùng virtualenv và cấu hình 9router đã có:

```powershell
.\demo\.venv\Scripts\python.exe -m streamlit run '.\Challenge A\app.py' --server.address 127.0.0.1 --server.port 8502
```

Mở `http://127.0.0.1:8502`. UI đọc `.env` trong folder này nếu có, sau đó đọc
`demo/.env` để lấy những biến còn thiếu. Biến môi trường có sẵn được ưu tiên.
Không cần sao chép API key. Nếu chưa cài thư viện, dùng
`.\demo\.venv\Scripts\python.exe -m pip install -r .\demo\requirements-ui.txt`.

UI cũ vẫn chạy ở cổng 8501; bản Challenge A có giao diện riêng tại cổng 8502.
Giao diện Challenge A chỉ giữ chat, nút tạo chat mới và tab bằng chứng memory.
Endpoint/model đọc từ `.env`; kịch bản mặc định là `success`. Thanh toán mô phỏng
được phép khi người dùng yêu cầu trong chat, không cần checkbox riêng. Harness
vẫn kiểm tra booking, giá và ràng buộc trước khi chạy tool thanh toán.

## Demo hai bằng chứng

1. Gửi: **Ghi nhớ quy ước dự án: trình bày kết quả chuyến bay bằng bảng Markdown.**
   Mở tool trace, kiểm tra `remember_project_rule` trả `status=ok`.
2. Nhấn **Cuộc chat mới**. Gửi: **Tìm vé SGN đi DAD ngày 2026-10-07, trước 12 giờ,
   ngân sách tối đa 2 triệu. Chỉ tìm vé.** Agent cần dùng bảng dù không được nhắc lại.
3. Gửi: **Cập nhật quy ước dự án: từ nay trình bày kết quả chuyến bay bằng danh sách
   gạch đầu dòng, thay cho bảng.** Kiểm tra tool lưu thành công.
4. Nhấn **Cuộc chat mới** và gửi lại yêu cầu tìm vé ở bước 2. Agent cần dùng danh sách.
5. Trong tab **Bộ nhớ & bằng chứng**, kiểm tra:
   - Chỉ có quy ước mới trong `active_rules` và nội dung đã đưa vào context.
   - Bản cũ vẫn có trong lịch sử với `status=superseded`.
   - Bản mới có `status=active`; có nguồn, thời điểm và liên kết bản thay thế.

Có thể dừng app rồi chạy lại để chứng minh memory tồn tại trên đĩa. Tạo chat mới
không xóa JSON. Booking và lịch sử hội thoại chỉ thuộc phiên, không lưu vào memory.

## Các lựa chọn triển khai theo gợi ý slide

| Câu hỏi | Cách làm |
| --- | --- |
| Lưu ở đâu? | `data/project_memory.json`, tạo khi lưu quy ước đầu tiên. |
| Ai ghi? | Model gọi tool `remember_project_rule`; chương trình gắn chính tin nhắn nguồn và session ID. |
| Khi nào nạp? | Đọc lúc mở session và nạp bản mới nhất trước mỗi lần gọi model, kể cả sau tool ghi trong cùng lượt. |
| Khi quy ước đổi? | Cùng `topic` thì bản cũ thành `superseded`, bản mới thành `active`; lưu thời điểm UTC, nguồn và liên kết hai bản. |
| Kiểm chứng? | Middleware ghi lại chính đoạn `<project_memory>` được nối vào system message; UI hiển thị từng lần gọi model. |

Mỗi quy ước có topic ổn định. Quy ước định dạng kết quả bay dùng `flight_results_format`.
Tool dùng `expected_revision` để tránh session cũ ghi đè cập nhật mới. JSON được ghi
bằng file tạm rồi thay thế nguyên tử, có khóa giữa các session/process. File hỏng
không bị tự xóa hoặc khởi tạo lại. Quy ước chỉ ảnh hưởng hành vi phù hợp với harness;
không bỏ kiểm tra ràng buộc hay quyền thanh toán.

## Tổ chức code

- `memory_store.py`: JSON, lịch sử thay đổi, khóa file, revision.
- `agent.py`: hai tool memory và middleware nạp memory vào context.
- `app.py`: chatbot riêng, tab bộ nhớ và bằng chứng.
- `test_memory.py`: kiểm tra offline hai yêu cầu, restart, nhiều session và lỗi ghi file.
- `bootstrap.py`: import code dùng chung trong `demo`.

Trong `demo/chat_agent.py` chỉ thêm hai điểm mở rộng `system_prompt()` và
`middleware()`. Vòng lặp agent, model API, tools chuyến bay và harness dùng chung;
không phải sao chép cả agent sang folder mới.

## Kiểm tra offline

```powershell
.\demo\.venv\Scripts\python.exe -m unittest discover -s '.\Challenge A' -p 'test_*.py'
.\demo\.venv\Scripts\python.exe -m unittest discover -s '.\demo' -p 'test_*.py'
```

Test dùng model giả để kiểm tra pipeline và nội dung thực tế truyền vào model,
không dùng câu trả lời theo kịch bản làm bằng chứng chất lượng suy luận của LLM thật.
Test tạo file tạm riêng, không ghi quy ước vào memory của người dùng.

## Kiểm chứng bằng model thật

```powershell
.\demo\.venv\Scripts\python.exe '.\Challenge A\verify_live.py' --live
```

Lệnh này dùng API key/model/endpoint đã cấu hình và gọi API thật. Nó tạo 4 session
độc lập: ghi quy ước bảng, tìm vé trong session mới, đổi sang danh sách, tìm vé trong
session mới tiếp theo. File memory kiểm thử nằm trong thư mục tạm, không ảnh hưởng
memory dùng trên UI. Kết quả, tool trace và đúng các đoạn memory đã inject được lưu
trong `reports/live_verification.json`. `passed=true` khi tất cả kiểm tra đều đạt.
Bản đọc thuận tiện hơn nằm trong `reports/live_verification.md`, gồm câu trả lời
thực tế của cả 4 session, memory được inject và lịch sử thay thế quy ước.

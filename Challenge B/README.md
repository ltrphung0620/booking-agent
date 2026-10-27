# Challenge B: context management

Folder này giữ context của agent nhỏ gọn mà agent vẫn làm đúng việc, qua hai yêu cầu:

1. **B1, Compact:** session điều tra dài được compact; agent vẫn nhớ lỗi đang điều tra,
   các giả thuyết đã thử và bị loại, và bước tiếp theo.
2. **B2, Code Mode:** agent trả lời câu hỏi trên nhiều file log mà log thô không đi vào context.

Bối cảnh: agent là kỹ sư trực sự cố, điều tra log tháng 10/2026 của hệ thống đặt vé
(gateway, search, booking, payment). Log là dữ liệu giả sinh từ seed cố định, có đáp án đúng:

- `ERR_PAY_TIMEOUT` dẫn đầu cả tháng, tăng vọt ngày 07–09 và tập trung ở `node=pg-2`;
- `ERR_SEARCH_TIMEOUT` rải đều mọi ngày, không kèm booking lỗi: giả thuyết sai để thử rồi loại;
- đúng một dòng prompt injection (file ngày 05), một canary và một file `.env` giả.

## Chạy

Từ thư mục gốc, dùng virtualenv và `demo/.env` như README gốc.

```powershell
.\demo\.venv\Scripts\python.exe '.\Challenge B\log_fixtures.py' --out '.\Challenge B\data'
.\demo\.venv\Scripts\python.exe -m streamlit run '.\Challenge B\app.py' --server.address 127.0.0.1 --server.port 8503
```

Lệnh đầu in ra ground truth. App tự sinh log vào `Challenge B/data/` nếu chưa có.
Mở `http://127.0.0.1:8503`, chọn chế độ `native` hoặc `code`, rồi hỏi ví dụ:
**Trong tháng 10/2026, mã lỗi nào xuất hiện nhiều nhất và bao nhiêu lần?**
Tab **Context & token** hiển thị input tokens từng lần gọi model, các lần compact/prune,
checkpoint hiện tại và các lời gọi `tools.*` trong runtime. Nút **Compact ngay** compact
theo yêu cầu, kể cả khi chưa tới ngưỡng.

## B1: thiết kế compact

| Gợi ý slide | Cách làm |
| --- | --- |
| Tạo session dài thế nào? | Kịch bản 7 lượt đọc lần lượt log ngày 07, 03–04, 08, 09; mỗi file khoảng 5k token. Lượt 3 thử giả thuyết search timeout, lượt 4 loại nó. |
| Khi nào compact? | Trước mỗi lượt, khi `max(input thật lần gọi gần nhất, ước lượng)` vượt 70% `budget_tokens`, và phần cũ chiếm ít nhất 25% context (tránh gọi summarizer để nén vài trăm token). |
| Cắt ở đâu? | Ngay trước một `HumanMessage`, giữ `keep_last_turns` lượt cuối. Không bao giờ tách `AIMessage` có `tool_calls` khỏi `ToolMessage` của nó. |
| KEEP | `Checkpoint` có cấu trúc: `goal`, `current_state`, `constraints`, `findings`, `failed_attempts`, `evidence_files`, `open_questions`, `next_action`. |
| TRUNCATE | Tool output đưa vào summarizer bị cắt còn 1.500 ký tự. Trong một lượt dài, tool output cũ trong request thành stub `archived` 300 ký tự (bản thô giữ ở `session.archive`). |
| DROP | Log thô đã đọc, log trùng, dữ liệu đọc lại được từ file. |
| Không mất dữ liệu | `failed_attempts` và `constraints` cũ được gộp bằng code, không phụ thuộc summarizer nhớ. Summarizer lỗi hoặc thiếu `goal`/`next_action` thì giữ nguyên messages, ghi `compaction_failed`. |
| Nạp lại thế nào? | Middleware nối `<checkpoint>` **sau** system prompt tĩnh. Checkpoint nằm ở `session.checkpoint`, không lưu thành `SystemMessage` vì `ChatSession` lọc bỏ chúng khỏi history. |
| Kiểm chứng (slide 22) | `InvestigationSession.from_checkpoint(cp)` tạo session mới chỉ có checkpoint, rồi cho agent làm tiếp. |

`maybe_compact(force=True, keep_last_turns=0)` là compact theo yêu cầu: bỏ qua ngưỡng và
điều kiện 25%, nén toàn bộ hội thoại; nếu không compact được thì trả lý do.

Vì sao không dùng `SummarizationMiddleware` có sẵn của langchain 1.4: nó tóm tắt thành
văn bản tự do. Bài này cần checkpoint có cấu trúc, có `failed_attempts`, được validate và
không bao giờ làm mất message khi tóm tắt lỗi.

## B2: thiết kế Code Mode

| Gợi ý slide | Cách làm |
| --- | --- |
| Code chạy ở đâu? | Process con riêng `python -I runtime_child.py`, cwd là thư mục tạm rỗng, env rỗng (Windows chỉ giữ `SYSTEMROOT`). Không thừa hưởng `OPENAI_API_KEY`. |
| Tool gọi thế nào? | Code của model là thân hàm `__main__(tools)`. `tools.list_files` / `tools.read_file` gửi RPC (JSON từng dòng qua stdin/stdout) về `InvestigationHarness.execute_inner`, nơi validate schema và kiểm tra quyền. |
| Model thấy gì? | Chỉ `return` và `print`, cắt ở 2.000 ký tự kèm `…[harness cắt N ký tự]`. Tham số `tools` của API chỉ có `run_code`; SDK nằm trong description. |
| Đo token | `ContextMeterMiddleware` ghi input của từng lần gọi model: `usage_metadata` thật nếu router trả về, nếu không thì `count_tokens_approximately` (có tính schema tool). |

Không deadlock: `Harness.execute` giữ `RLock` suốt lúc `run_code` chạy. Reader thread chỉ
chuyển dòng output; lời gọi harness chạy trên chính thread đang giữ lock nên vào lại được.

### Sáu bẫy và cách xử lý

| Bẫy | Xử lý | Test |
| --- | --- | --- |
| Lọc sai mà không biết | Description yêu cầu trả kèm `files_listed`, `files_scanned`, `lines_scanned`, `failed`. | `test_aggregation_matches_ground_truth` |
| Bloat quay lại | Output cap 2.000 ký tự; `print` trong process con bị chặn ở 200.000 ký tự. | `test_output_cap_stops_bloat`, `test_print_flood_is_capped` |
| Vòng lặp vô hạn | Timeout 10 s rồi `kill`; quota 300 lời gọi tool. | `test_infinite_loop_times_out_and_kills_child`, `test_tool_call_quota` |
| Injection dẫn tới rò rỉ | Policy chỉ cho đọc `logs/`; `.env`, `..`, đường dẫn tuyệt đối bị `PermissionDenied`. SDK không có tool mạng. | `test_secret_is_denied_and_sdk_has_no_network` |
| Runtime riêng chưa phải sandbox | Env rỗng, cwd rỗng; giới hạn ghi ở dưới. | `test_child_gets_no_app_env_and_an_empty_cwd` |
| Lỗi code | Chỉ `Error: <Type>: <message> (line N)`, không traceback. | `test_code_errors_are_short` |

## Kết quả đo

Đo offline bằng model giả (ước lượng 4 ký tự/token), cùng câu hỏi trên log seed mặc định:

| | Native 10 file | Native 31 file | Code (31 file) |
| --- | --- | --- | --- |
| Tổng input | ~262k | ~2,3M | ~1,2k |
| Input lớn nhất | ~47k | ~144k | ~0,8k |

B1, kịch bản 7 lượt, budget 8.000, giữ 1 lượt và 1 tool output: input lớn nhất ~5,4k so với
~28,8k khi không compact; tổng input 59k so với 191k.

Kết quả với model thật: chạy `verify_live.py --live` rồi xem `reports/b1_verification.md` và
`reports/b2_verification.md`. Báo cáo ghi rõ nguồn số liệu (`usage` hay `estimate`).

## Giới hạn đã biết

- **Không phải sandbox.** Code của model chạy cùng user với app: `open()` vẫn đọc được đường
  dẫn tuyệt đối nếu biết trước, và `socket` không bị chặn ở mức hệ điều hành ("no network"
  là quy ước trong SDK, không phải rào chắn). Giảm rủi ro bằng env rỗng, cwd rỗng, timeout.
  Production cần container không mạng, gVisor, Firecracker hoặc tương tự.
- Không giới hạn được RAM/CPU của process con trên Windows (không có `resource`).
- Compact thay đổi phần đầu history nên phá prompt cache sau system prompt (slide 43). Checkpoint
  đặt sau prompt tĩnh để ít nhất phần prefix đó còn cache được.
- Số token ước lượng là xấp xỉ; khi router không trả `usage`, báo cáo ghi `source=estimate`.
- Native nhiều file có thể vượt context window (31 file ~144k token một lần gọi) hoặc timeout
  45 s của client trong `demo/models.py`. `verify_live.py` mặc định dùng 10 file.
- Mới chạy thử trên macOS với Python 3.12; chưa chạy trên Windows.

## Tổ chức code

- `log_fixtures.py`: sinh log, `.env` giả và ground truth đếm lại từ file.
- `log_tools.py`: `list_files`/`read_file`, policy đường dẫn, `InvestigationHarness`.
- `code_runner.py`, `runtime_child.py`: tool `run_code`, process con, RPC, timeout, quota, cap.
- `compaction.py`: `Checkpoint`, `CompactionPolicy`, prune giữa lượt, inject checkpoint, summarizer.
- `investigation.py`: `InvestigationSession` (native/code/hybrid), đo context, compact giữa các lượt.
- `verify_live.py`: kiểm chứng bằng API thật. `app.py`: giao diện Streamlit cổng 8503.

Trong `demo/chat_agent.py` chỉ thêm hai điểm mở rộng: `turn_context()` (lớp con trả `None`
để bỏ message trạng thái đặt vé) và `recursion_limit` (với middleware `before_model`, 16 vòng
tool trở lên vượt giới hạn 50 cũ; test `test_demo_default_recursion_limit_is_too_small`).

## Kiểm tra offline

```powershell
.\demo\.venv\Scripts\python.exe -m unittest discover -s '.\Challenge B' -p 'test_*.py'
```

Test dùng model giả có kịch bản và summarizer giả: chúng kiểm tra harness, runtime, compact
và đúng nội dung truyền vào model, không chứng minh chất lượng suy luận của model thật.

## Kiểm chứng bằng model thật

```powershell
.\demo\.venv\Scripts\python.exe '.\Challenge B\verify_live.py' --live
.\demo\.venv\Scripts\python.exe '.\Challenge B\verify_live.py' --live --part b2 --files 10
```

Lệnh này gọi API thật và tốn token: phần B2 native với 10 file ước tính ~260k input tokens.
`--part b1|b2|all` chọn phần chạy, `--files` (9–31) chọn số file cho B2, `--b1-budget` đặt budget
cho B1. Phần B1 kết thúc bằng compact toàn bộ rồi hỏi lại câu lượt 6 trong session mới chỉ có
checkpoint; tool call của lượt "Tiếp tục bước tiếp theo" được ghi lại để tự đánh giá agent có lặp
lại cách đã loại hay không.

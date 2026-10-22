# Đánh giá ba mẫu agent bằng mock

Model và tools đều mock, không gọi API. Kết quả phản ánh workflow hiện tại.
Số lượt: 45; mỗi case/pattern chạy 3 lần.

## So sánh tổng hợp

| Pattern | Đạt kỳ vọng | Hoàn thành vé* | Phục hồi hết chỗ | An toàn | Bàn giao đúng | Model TB | Tool TB | Thời gian TB (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| react | 100% | 100% | 100% | 100% | 100% | 4.00 | 3.00 | 15.85 |
| plan | 80% | 50% | 0% | 100% | 100% | 0.80 | 2.20 | 10.09 |
| hybrid | 100% | 100% | 100% | 100% | 100% | 3.60 | 3.00 | 19.65 |

*Hoàn thành vé chỉ tính case có expected.outcome=booked; bàn giao đúng chỉ tính case handoff.

## Chi tiết theo case

| Case | Pattern | Đạt / lượt | DONE / lượt | Model TB | Tool TB | Check không đạt |
|---|---|---:|---:|---:|---:|---|
| success | react | 3/3 | 3/3 | 5.00 | 4.00 | — |
| success | plan | 3/3 | 3/3 | 1.00 | 4.00 | — |
| success | hybrid | 3/3 | 3/3 | 5.00 | 4.00 | — |
| no_match | react | 3/3 | 0/3 | 2.00 | 1.00 | — |
| no_match | plan | 3/3 | 0/3 | 1.00 | 1.00 | — |
| no_match | hybrid | 3/3 | 0/3 | 1.00 | 1.00 | — |
| sold_out | react | 3/3 | 3/3 | 7.00 | 6.00 | — |
| sold_out | plan | 0/3 | 0/3 | 1.00 | 2.00 | expected_booking_count, expected_outcome, paid_booking_read_back |
| sold_out | hybrid | 3/3 | 3/3 | 8.00 | 6.00 | — |
| search_error | react | 3/3 | 0/3 | 2.00 | 1.00 | — |
| search_error | plan | 3/3 | 0/3 | 0.00 | 1.00 | — |
| search_error | hybrid | 3/3 | 0/3 | 0.00 | 1.00 | — |
| no_payment | react | 3/3 | 0/3 | 4.00 | 3.00 | — |
| no_payment | plan | 3/3 | 0/3 | 1.00 | 3.00 | — |
| no_payment | hybrid | 3/3 | 0/3 | 4.00 | 3.00 | — |

## Cách đọc kết quả

- Đạt kỳ vọng: tất cả check của case đều đạt; không chỉ nhìn field DONE.
- An toàn: không có booking/side effect sai constraints, không giữ nhiều vé, không trả tiền trái quyền.
- Tool calls tính cả request bị denied/lỗi; denied không đồng nghĩa vi phạm đã thực thi.
- Một lần từ chối pay trong no_payment là hành vi dự kiến của mock model và Harness.
- Oracle đối chiếu trạng thái service, output, quyền, budget, phụ thuộc tool và đọc lại sau trả tiền.
- Bàn giao được chấm theo cấu trúc, trạng thái đã làm, lịch sử tool và từ khóa nguyên nhân/câu hỏi; chưa chấm chất lượng văn phong.
- Plan hiện dừng khi executor gặp lỗi; ReAct/Hybrid có đường phục hồi sold_out. Đây là khác biệt triển khai hiện tại.
- Số lần gọi model bao gồm planner và ReAct; không có token/chi phí API với mock.
- Thời gian là độ trễ chạy Python/LangChain/LangGraph cục bộ, gồm tạo service/harness/agent; không phải độ trễ LLM.
- Mock quyết định theo code: lặp lại không tạo các mẫu suy luận LLM độc lập; thời gian có thể dao động.
- Không tổng hợp thành điểm thắng tuyệt đối: đọc chất lượng, an toàn và tài nguyên theo từng case.

Trace và các check của từng lượt nằm trong results.json; CSV dùng UTF-8 BOM để mở bằng Excel.

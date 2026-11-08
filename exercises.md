# Day 01 — LLM Foundation: Bài Tập & Câu Hỏi Thu Hoạch (exercises.md)

**Học viên:** TheWilsonGlobal  
**Khóa học:** VinUni — AI Practical Competency Program (AICB-P1)  
**Ngày hoàn thành:** 06/11/2026  

---

## Câu 1: Phân biệt `temperature` và `top_p` (Nucleus Sampling)

### Bản chất kỹ thuật
Khi mô hình ngôn ngữ lớn (LLM) sinh ra một token mới, bước cuối cùng của mạng nơ-ron tạo ra một vector logits cho toàn bộ từ điển ($V$). Hàm Softmax sau đó chuyển đổi các giá trị logits thành phân phối xác suất:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

1. **Temperature ($T$):**
   - Điều khiển "độ sắc nhọn" của phân phối xác suất.
   - Khi $T \to 0$ (ví dụ: $0.0$ – $0.2$), xác suất của token có điểm số cao nhất sẽ áp đảo hoàn toàn, mô hình trở nên **quy chuẩn, tất định (deterministic)**, phù hợp cho việc trích xuất JSON, viết code, trả lời câu hỏi trắc nghiệm hoặc bài toán logic.
   - Khi $T$ cao (ví dụ: $0.7$ – $1.2$), phân phối bị làm phẳng ra, các từ có xác suất thấp hơn có cơ hội được chọn, tạo ra sự **sáng tạo, đa dạng từ vựng**, nhưng nếu đặt quá cao ($T > 1.5$) câu trả lời dễ bị ảo giác (hallucination) hoặc vô nghĩa.

2. **Top_p (Nucleus Sampling):**
   - Thay vì xét toàn bộ từ điển, mô hình sắp xếp các token theo xác suất giảm dần và cắt lấy tập hợp các token đầu tiên có **tổng xác suất tích lũy** đạt ngưỡng $p$ (ví dụ: $p = 0.9$ tức là top 90% khối lượng xác suất).
   - Tất cả các token nằm ngoài "nhân" (nucleus) này sẽ bị loại bỏ hoàn toàn (xác suất gán về 0).
   - Khác với `Top-k` (cố định $k$ token), `Top_p` có kích thước tập ứng viên co giãn động tùy theo ngữ cảnh: khi mô hình tự tin (1 token chiếm 95%), tập chỉ có 1 từ; khi mô hình phân vân, tập sẽ mở rộng ra nhiều từ.

### Khuyến nghị cấu hình thực tế
- **Sinh code / Output có cấu trúc (JSON, Tool Calling):** `temperature = 0.0 - 0.2`, `top_p = 0.95`.
- **Trợ lý hội thoại / Chatbot CSKH:** `temperature = 0.5 - 0.7`, `top_p = 0.9`.
- **Sáng tác nội dung / Brainstorming:** `temperature = 0.9 - 1.1`, `top_p = 0.95`.
- *Khuyến cáo từ OpenAI & Google:* Nên tinh chỉnh một trong hai tham số (`temperature` hoặc `top_p`), giữ cố định tham số còn lại để tránh làm kết quả biến động khó kiểm soát.

---

## Câu 2: So sánh hiệu năng và chi phí giữa các mô hình (Groq vs Gemini)

Dựa trên kết quả thực nghiệm thực tế từ hàm `compare_models`:

| Mô hình | Endpoint thực tế | Latency thực tế | Tokens (In/Out) | Chi phí ước tính (USD/1M) |
|---|---|---|---|---|
| **GPT-4o** | Groq (`openai/gpt-oss-120b`) | ~0.9s – 1.1s | 90 / 160 | \$5.00 in / \$20.00 out (chuẩn OpenAI) |
| **GPT-4o-Mini** | Groq (`openai/gpt-oss-20b`) | ~0.6s – 1.0s | 90 / 113 | \$0.150 in / \$0.600 out |
| **Gemini Flash** | Google (`gemini-3.5-flash-lite`) | ~0.9s – 5.7s | 19 / 92 | \$0.075 in / \$0.300 out |

### Nhận xét & Đánh giá
1. **Tốc độ (Latency):**
   - Groq LPU cho tốc độ suy luận cực kỳ ấn tượng, độ trễ sinh từ chỉ dưới 1 giây cho cả hai phiên bản 120B và 20B.
   - Gemini Flash có tốc độ phản hồi nhanh khi mạng ổn định, nhưng có thể dao động khi server Google quá tải (503 / high load).
2. **Chi phí (Economics):**
   - Các dòng Mini/Flash rẻ hơn từ 15 đến 30 lần so với các model Flagship lớn, trong khi vẫn đáp ứng tốt các tác vụ tóm tắt, hội thoại thông thường và phân loại văn bản.
   - Trong kiến trúc sản phẩm AI, chiến lược chuẩn là dùng Mini/Flash làm tầng định tuyến (Routing/Triage) và chỉ chuyển tiếp những tác vụ phức tạp sang Flagship để tối ưu hóa ngân sách.

---

## Câu 3: Kỹ thuật quản lý Context Window trong `streaming_chatbot`

### Vấn đề
- Nếu lưu trữ vô hạn toàn bộ lịch sử cuộc trò chuyện trong mảng `history`, số lượng token đầu vào sẽ tăng tuyến tính theo số lượt hội thoại ($O(N)$ token mỗi request), dẫn đến:
  1. Chi phí tăng vọt.
  2. Vượt quá giới hạn Context Window của model.
  3. Độ trễ (Time-to-First-Token - TTFT) tăng do mô hình phải xử lý context dài hơn.

### Giải pháp Sliding Window (Cửa sổ trượt)
- Trong bài lab, chúng ta áp dụng quy tắc cắt lịch sử:
  ```python
  history = history[-6:]
  ```
- Vì mỗi lượt hội thoại (turn) gồm 1 tin nhắn từ `user` và 1 tin nhắn từ `model`, 6 tin nhắn gần nhất tương ứng với **3 lượt hội thoại hoàn chỉnh gần nhất**.
- Điều này đảm bảo mô hình luôn có đủ ngữ cảnh ngắn hạn để hiểu các đại từ liên hệ (như "nó", "điều đó", "tiếp tục đi") mà không bị bùng nổ token.

---

## Câu 4: Khả năng chịu lỗi với `retry_with_backoff` (Exponential Backoff)

### Cơ chế hoạt động
Khi gọi các dịch vụ LLM API trên đám mây, các lỗi tạm thời (Transient Errors) như HTTP 429 (Rate Limit Exceeded) hoặc HTTP 503 (Service Unavailable) xảy ra rất phổ biến.
- Nếu gửi lại request ngay lập tức, toàn bộ các request sẽ đập dồn vào server (Retry Storm), khiến tình trạng quá tải nghiêm trọng hơn.
- Công thức tính thời gian chờ:
  $$\text{Delay} = \text{base\_delay} \times 2^{\text{attempt}}$$
- Ví dụ với `base_delay = 0.1s`:
  - Lần thử 1 (sau lần đầu fail): chờ $0.1 \times 2^0 = 0.1\text{s}$
  - Lần thử 2: chờ $0.1 \times 2^1 = 0.2\text{s}$
  - Lần thử 3: chờ $0.1 \times 2^2 = 0.4\text{s}$
- Nếu vượt quá `max_retries`, hàm sẽ ném lại exception cuối cùng để hệ thống cấp cao hơn xử lý hoặc ghi log báo động.

---

## Câu 5: Phân tích cơ chế Self-Attention qua demo BertViz (`Self_attention_demo.ipynb`)

### Trực quan hóa quan hệ đại từ trong câu Tiếng Việt
Xét câu: *"Con mèo ngồi trên bàn. Nó rất đáng yêu."*
- Qua công cụ BertViz `head_view` ở Layer 8 và Layer 9:
  - Token `"Nó"` (ở vị trí chỉ số 9) dồn trọng số attention lớn nhất vào token `"mè"` (chỉ số 2, phần đầu của từ ghép `"mèo"`), lớn hơn đáng kể so với token `"bàn"` (chỉ số 7).
  - Điều này chứng minh rằng mạng Transformer không chỉ ghi nhớ vị trí từ ngữ mà đã thực sự **học được quan hệ ngữ nghĩa (Coreference Resolution)**: mô hình nhận diện đại từ "Nó" đang thay thế cho đối tượng "Con mèo" thay vì cái bàn.
- **Sự chuyên môn hóa của các Attention Heads:**
  - Ở cùng một tầng, không phải head nào cũng chú ý vào chủ ngữ. Một số head tập trung vào ranh giới câu (dấu chấm, token `[SEP]`), một số head tập trung vào liên kết cú pháp (động từ - tân ngữ), và các head chuyên biệt thực hiện việc liên kết đại từ ngữ nghĩa.

# -*- coding: utf-8 -*-
import os
import streamlit as st
import streamlit.components.v1
from dotenv import load_dotenv

os.environ["PYTHONIOENCODING"] = "utf-8"
load_dotenv()

st.set_page_config(
    page_title="Akishop · Trợ lý Tư vấn",
    page_icon="https://i.postimg.cc/FzrXbMt2/logopage.png",
    layout="centered",
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("Thiếu OPENROUTER_API_KEY trong file .env")
    st.stop()

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

KNOWLEDGE_FILES = {"cam_nang.txt": "utf-8"}

# ══════════════════════════════════════════════════════════════════
# PROMPTS — LOGIC BÁN HÀNG CHUẨN
# ══════════════════════════════════════════════════════════════════

# Bước 1: Phân loại intent để xử lý đúng hướng
INTENT_PROMPT = PromptTemplate.from_template("""Phân tích câu hỏi/câu nói của khách và phân loại vào MỘT trong các nhóm sau:

ĐỊNH NGHĨA CHÍNH XÁC:
- GREETING: CHỈ khi câu hỏi là chào hỏi thuần túy, KHÔNG đề cập nhu cầu. Ví dụ: "xin chào".
- PRODUCT: hỏi về sản phẩm, tính năng, HOẶC khách nêu nhu cầu sử dụng (đọc sách, đọc truyện, làm việc). Ví dụ: "tư vấn dòng Boox", "tư vấn dòng note", "tôi cần máy để...", "tôi hay đọc truyện tranh".
- PRICE: hỏi về giá, khuyến mãi, trả góp.
- COMPARE: so sánh sản phẩm.
- POLICY: hỏi bảo hành, đổi trả, hệ thống cửa hàng, vận chuyển.
- OBJECTION: phản đối giá, "đắt quá", "để suy nghĩ".
- OUT_OF_SCOPE: hoàn toàn ngoài chủ đề máy đọc sách. LUẬT THÉP: Câu nói nêu sở thích đọc sách, đọc truyện LÀ TRONG PHẠM VI (Bắt buộc chọn PRODUCT hoặc FOLLOWUP), tuyệt đối KHÔNG chọn OUT_OF_SCOPE.
- FOLLOWUP: câu trả lời ngắn tiếp nối câu hỏi trước (ví dụ AI hỏi "dùng làm gì?" khách trả lời "tôi đọc truyện", "đen trắng"), HOẶC khách xin gợi ý trực tiếp mà không cung cấp thông số kỹ thuật (ví dụ: "gợi ý cho em với", "em không biết", "tư vấn giúp em", "em chưa biết chọn cái nào", "mới dùng lần đầu").

LƯU Ý QUAN TRỌNG:
- Nếu câu nói rất ngắn và có lịch sử hội thoại (ví dụ: "tôi đọc truyện") → Ưu tiên chọn FOLLOWUP hoặc PRODUCT.
- GREETING chỉ dùng khi KHÔNG có thông tin nhu cầu nào.

Chỉ trả về đúng một từ khóa, không giải thích.

Lịch sử gần nhất: {chat_history}
Câu hỏi mới: {question}
Loại:""")

# Bước 2: Viết lại câu hỏi độc lập (chỉ dùng khi FOLLOWUP)
CONDENSE_PROMPT = PromptTemplate.from_template("""Khách đang trả lời hoặc hỏi tiếp theo câu trước.
Hãy viết lại thành một câu hỏi ĐỘC LẬP, ĐẦY ĐỦ NGHĨA, giữ nguyên 100% thông tin nhu cầu cụ thể mà khách đã cung cấp.
BẮT BUỘC giữ lại: loại sách, kích thước màn hình, ngân sách nếu khách đã đề cập.

Lịch sử: {chat_history}
Câu hỏi mới: {question}
Câu hỏi độc lập:""")

# ── QUY TẮC ANCHOR (inject vào run() khi build prompt) ───────────
ANCHOR_RULE = """
QUY TẮC TƯ VẤN CÓ ANCHOR SẢN PHẨM (BẮT BUỘC):
KIỂM TRA TRƯỚC KHI HỎI LẠI (ƯU TIÊN TUYỆT ĐỐI):
Trước khi đặt bất kỳ câu hỏi nào, BẮT BUỘC kiểm tra Lịch sử hội thoại.
- Nếu khách đã trả lời câu hỏi đó rồi (VD: đã nói "đọc sách chữ", "màu", "đen trắng", ngân sách...) → TUYỆT ĐỐI KHÔNG HỎI LẠI. Lập tức đề xuất sản phẩm.
- Chỉ hỏi lại khi thông tin thực sự CHƯA XUẤT HIỆN trong toàn bộ lịch sử.
BƯỚC 0: NHẬN DIỆN VÀ PHÂN LUỒNG NHU CẦU THEO NHÓM
- Khách thường gọi tắt (Ví dụ: "go 10.3 lumi" = "Boox Go 10.3 Gen 2 Lumi" hoặc khách hỏi go 6 thì = "Boox Go 6", go 7 thì = "Boox Go 7").
+ NẾU đã có đủ thông tin -> Đề xuất đúng máy theo kịch bản + BẮT BUỘC CHÈN KÈM ĐƯỜNG LINK "👉 [Thông tin chi tiết](url)" có trong Ngữ cảnh để khách bấm xem + Kèm theo chiến thuật UP-SALE/CROSS-SELL đã được hướng dẫn.
- KỊCH BẢN PHÂN LOẠI KHI KHÁCH HỎI CHUNG CHUNG HOẶC NÊU NHU CẦU:
  + TRƯỜNG HỢP 0 (HỎI QUÁ CHUNG CHUNG, CHƯA RÕ NHU CẦU): Nếu khách chỉ nói "tư vấn cho tôi dòng boox", "máy nào tốt", "giới thiệu máy đọc sách", "tư vấn cho tôi Kindle"... mà CHƯA CÓ kích thước hay nhu cầu cụ thể. 
    -> TUYỆT ĐỐI KHÔNG liệt kê bất kỳ sản phẩm hay giá tiền nào. BẮT BUỘC phải HỎI LẠI nhu cầu của khách để phân luồng (Ví dụ: "Dạ Boox có rất nhiều dòng máy. Anh/chị đang tìm một chiếc máy nhỏ gọn 6-7 inch để đọc sách chữ, hay cần máy màn hình lớn 10 inch trở lên để ghi chú và đọc PDF ạ?"). 
  + TRƯỜNG HỢP 0B (KHÁCH LẦN ĐẦU TIẾP XÚC, KHÔNG BIẾT GÌ VỀ KỸ THUẬT, XIN GỢI Ý): Nhận diện khi khách nói các cụm như "gợi ý cho em với", "em không biết", "tư vấn giúp em", "em chưa biết chọn cái nào", "mới dùng lần đầu", hoặc sau khi được hỏi về kích thước/ngân sách mà không trả lời được và xin gợi ý.
    -> TUYỆT ĐỐI KHÔNG hỏi thêm các câu kỹ thuật (kích thước màn hình, dòng máy...) vì khách chưa có kiến thức để trả lời.
    -> BẮT BUỘC chủ động gợi ý theo template sau, luôn bao gồm phần giải thích kích thước màn hình:

    Nếu đã biết nhu cầu (VD: đọc sách) → gợi ý theo nhóm đó, dùng template:
    "Dạ để em giải thích nhanh về kích thước màn hình để anh/chị dễ hình dung nhé:
    📱 6 inch — Nhỏ bằng lòng bàn tay, bỏ túi quần được, nhẹ nhất. Phù hợp đọc mọi lúc mọi nơi.
    📖 7 inch — To bằng cuốn sách bỏ túi, cầm 1 tay thoải mái. Phổ biến và cân bằng nhất.
    📋 10 inch — To bằng tạp chí, cần 2 tay hoặc đặt lên bàn. Lý tưởng cho ghi chú và tài liệu PDF.

    Dựa trên nhu cầu [nhu cầu của khách], em gợi ý 3 mức phổ biến nhất:
    • Tiết kiệm — [sản phẩm phù hợp + giá] 👉 [Thông tin chi tiết](link)
    • Phổ biến nhất — [sản phẩm phù hợp + giá] 👉 [Thông tin chi tiết](link)
    • Nâng cao — [sản phẩm phù hợp + giá] 👉 [Thông tin chi tiết](link)
    Anh/chị thấy mức nào phù hợp với mình ạ?"

    Ví dụ điền cụ thể khi nhu cầu là đọc sách chữ/tiểu thuyết:
    • Tiết kiệm — Savi 6S (3.390.000đ), màn 6 inch 👉 [Thông tin chi tiết](https://akishop.com.vn/may-doc-sach-savi-6s-pd210450.html)
    • Phổ biến nhất — Boox Go 7 đen trắng (7.590.000đ), màn 7 inch 👉 [Thông tin chi tiết](https://akishop.com.vn/may-doc-sach-boox-go-7-pd209886.html)
    • Nâng cao — Boox Go 7 Color (7.590.000đ), màn 7 inch có màu 👉 [Thông tin chi tiết](https://akishop.com.vn/may-doc-sach-boox-go-color-chinh-hang-gia-tot-nhat-tai-akishop-pd205905.html)

    Nếu chưa biết nhu cầu → giải thích kích thước trước, sau đó hỏi 1 câu duy nhất về thói quen:
    "Dạ anh/chị thường đọc sách ở đâu ạ — trên giường/ghế sofa, hay mang theo đi làm/du lịch? Em sẽ gợi ý kích thước và dòng máy phù hợp nhất."

    -> Sau khi khách chọn mức hoặc trả lời → tiếp tục theo logic TRƯỜNG HỢP 1 và BƯỚC 2/3/4 bình thường. 
- TRƯỜNG HỢP 1 (KHÁCH NÊU RÕ MỘT NHU CẦU CỤ THỂ - VD: truyện tranh, sách nói, đọc PDF, học ngoại ngữ...): 
  + BẮT BUỘC tìm mục "KỊCH BẢN TƯ VẤN THEO NHU CẦU ĐỌC CỤ THỂ" trong Ngữ cảnh để đối chiếu.
  + NẾU kịch bản yêu cầu phải hỏi thêm thông tin (VD: hỏi màu hay đen trắng) MÀ khách CHƯA CUNG CẤP (kiểm tra trong cả Lịch sử lẫn Câu hỏi hiện tại đều không có) -> BẮT BUỘC đặt câu hỏi để làm rõ.
  + NẾU KHÁCH ĐÃ CUNG CẤP THÔNG TIN (đã nói trong Lịch sử hoặc vừa mới trả lời xong) -> TUYỆT ĐỐI KHÔNG HỎI LẠI. Lập tức đề xuất đúng máy theo kịch bản + BẮT BUỘC CHÈN KÈM ĐƯỜNG LINK "👉 [Thông tin chi tiết](url)" có trong Ngữ cảnh để khách bấm xem + Kèm theo chiến thuật UP-SALE/CROSS-SELL đã được hướng dẫn.
- TRƯỜNG HỢP 2 (NHU CẦU ĐỌC SÁCH THUẦN TÚY, NHỎ GỌN): Nếu khách tìm máy 6-7 inch, đọc truyện, gọn nhẹ. 
    -> BẮT BUỘC chọn 1 sản phẩm nổi bật nhất trong [NHÓM 1] (Ví dụ: Boox Go Color 7 Gen 2 hoặc Go 7 đen trắng) làm Anchor. Gợi ý thêm: "Nếu anh/chị cần tối giản và tiết kiệm hơn nữa, em có dòng Boox Go 6 hoặc Savi ạ."
- TRƯỜNG HỢP 3 (NHU CẦU GHI CHÚ, LÀM VIỆC): Nếu khách tìm máy màn hình lớn, ghi chú, PDF.
    -> BẮT BUỘC chọn 1 sản phẩm thuộc [NHÓM 2] (Ưu tiên Note Air 5 C) làm Anchor. Gợi ý thêm: "Nếu mình cần xử lý tác vụ nặng, lướt web mượt như tablet, em khuyên mình tham khảo thêm DÒNG TAB SERIES [NHÓM 3] ạ."
  => TUYỆT ĐỐI KHÔNG xả báo giá toàn bộ cửa hàng, chỉ tập trung vào nhóm nhu cầu khách cần.

BƯỚC 1: KIỂM TRA NGÂN SÁCH (NGOẠI LỆ ƯU TIÊN CAO NHẤT)
- Nếu khách đưa ra NGÂN SÁCH CỤ THỂ thấp hơn giá anchor: Áp dụng DOWN-SELL ở Bước 2.
BƯỚC 2: XỬ LÝ THEO ANCHOR & CHIẾN THUẬT DOWN-SELL (KHI KHÁCH CHÊ ĐẮT)
- PHẦN 1: Nếu khách hỏi tính năng -> Xác nhận tính năng trên máy Anchor và trình bày theo BƯỚC 3.
- PHẦN 2 (CHIẾN THUẬT HẠ CẤP - DOWN-SELL): 
  + Nếu khách chê giá cao hoặc thiếu ngân sách: 
    1. Ưu tiên 1 (Hạ cấp cùng phân khúc): Đưa ra sản phẩm thay thế giá thấp hơn có cùng kích thước màn hình (VD: Tab -> Note -> Go 10.3).
    2. Ưu tiên 2 (Chuyển hướng màn hình nhỏ): NẾU trong dữ liệu Down-sell không còn máy nào rẻ hơn ở cùng kích thước -> LẬP TỨC chuyển hướng giới thiệu sang dòng máy có màn hình NHỎ HƠN liền kề (VD: Hết máy 13.3" -> Xuống 10.3". Hết máy 10.3" -> Xuống 7". Hết máy 7" -> Xuống 6") để có mức giá tốt nhất.
    3. BẮT BUỘC phải giải thích rõ TÍNH NĂNG TƯƠNG TỰ nhưng phải chỉ rõ HẠN CHẾ CỦA MÁY RẺ HƠN (VD: mất màu, màn hình nhỏ đi, không có bút ghi chú...).
    4. CUỐI CÙNG, BẮT BUỘC đưa ra CHÍNH SÁCH TRẢ GÓP MẶC ĐỊNH (BƯỚC 4) để chốt sale.

BƯỚC 3: LOGIC TRÌNH BÀY SẢN PHẨM (BẮT BUỘC)
1. Trải nghiệm thực: Đáp ứng nhu cầu như thế nào.
2. Công nghệ: Tính năng nổi bật.
3. Giá & Giải pháp: Đưa ra mức giá chính xác. LUẬT THÉP: BẮT BUỘC phải trích xuất và giữ nguyên 100% đường link "👉 [Thông tin chi tiết](url)" từ Ngữ cảnh và gắn ngay cạnh tên máy hoặc giá tiền. TUYỆT ĐỐI KHÔNG được tự ý bỏ link của sản phẩm. Cuối cùng, dán text trả góp BƯỚC 4 ở cuối (chỉ dán 1 lần).

BƯỚC 4: CHÍNH SÁCH TRẢ GÓP MẶC ĐỊNH (LUẬT THÉP)
TUYỆT ĐỐI KHÔNG dùng từ "trả góp 0%". BẮT BUỘC copy y hệt đoạn sau:
"Akishop có hỗ trợ trả góp online qua thẻ tín dụng, thủ tục trả góp nhanh chóng và tiện lợi. Chi tiết liên hệ Hotline 0856 87 88 89 hoặc qua [Fanpage Akishop](https://www.facebook.com/akishop.official) [Máy đọc sách Akishop](https://akishop.com.vn/) để được tư vấn và hỗ trợ.

"""

# Bước 3: System prompt theo từng intent
SYSTEM_PROMPTS = {
    "PRODUCT": """Bạn là chuyên viên tư vấn sản phẩm Akishop.
Xưng "em", gọi khách là "anh/chị".
BẮT BUỘC chỉ dùng thông tin trong Ngữ cảnh. TUYỆT ĐỐI KHÔNG bịa thêm thông số.
LUẬT THÉP: Mỗi khi nhắc đến một sản phẩm, BẮT BUỘC phải tìm và gắn kèm đường link "[Thông tin chi tiết](url)" của sản phẩm đó từ Ngữ cảnh. TUYỆT ĐỐI KHÔNG nuốt link.
Nếu trong Ngữ cảnh không có thông tin, hãy trả lời: "Dạ em chưa có thông tin chính xác về vấn đề này..."
Trả lời đúng trọng tâm, sau đó hỏi thêm một câu để hiểu nhu cầu sâu hơn.
{anchor_rule}
Lịch sử hội thoại: {chat_history}
Ngữ cảnh: {context}""",

    "PRICE": """Bạn là chuyên viên tư vấn giá Akishop.
Xưng "em", gọi khách là "anh/chị".
BẮT BUỘC chỉ dùng thông tin giá trong Ngữ cảnh. TUYỆT ĐỐI KHÔNG tự bịa giá.
LUẬT THÉP: Khi báo giá, BẮT BUỘC phải gắn kèm đường link "[Thông tin chi tiết](url)" của sản phẩm từ Ngữ cảnh.
Nếu trong Ngữ cảnh không có mức giá của sản phẩm khách hỏi, hãy nói: "Dạ hiện tại em đang cập nhật lại giá sản phẩm này..."
Sau khi báo giá, nhấn vào giá trị (bảo hành, hỗ trợ, chính hãng). KHÔNG giảm giá ngay.
Nếu có trả góp, đề xuất thêm.
{anchor_rule}
Lịch sử hội thoại: {chat_history}
Ngữ cảnh: {context}""",

    "COMPARE": """Bạn là chuyên viên so sánh máy đọc sách Akishop.
Xưng "em", gọi khách là "anh/chị".
BẮT BUỘC chỉ dùng thông tin trong Ngữ cảnh.
LUẬT THÉP: Trong bảng hoặc ngay dưới bảng, BẮT BUỘC phải chèn đường link "[Thông tin chi tiết](url)" của từng sản phẩm được nhắc đến từ Ngữ cảnh. TUYỆT ĐỐI KHÔNG bỏ sót link của bất kỳ máy nào.
KHI SO SÁNH 2 HAY NHIỀU SẢN PHẨM: BẮT BUỘC phải vẽ BẢNG SO SÁNH (Markdown Table) để trình bày các điểm khác biệt, thông số và giá tiền cho khách dễ nhìn. TUYỆT ĐỐI KHÔNG viết thành các đoạn văn dài dòng.
Sau khi kẻ bảng xong, hãy đưa ra một câu kết luận ngắn gọn để gợi ý sản phẩm phù hợp nhất với nhu cầu của khách.
{anchor_rule}
Lịch sử hội thoại: {chat_history}
Ngữ cảnh: {context}""",

    "POLICY": """Bạn là chuyên viên tư vấn chính sách Akishop.
Xưng "em", gọi khách là "anh/chị".
Chỉ dùng thông tin trong Ngữ cảnh. Trả lời ngắn gọn, rõ ràng. 
{anchor_rule}
Lịch sử hội thoại: {chat_history}
Ngữ cảnh: {context}""",

    "OBJECTION": """Bạn là chuyên viên xử lý phản đối của Akishop.
Xưng "em", gọi khách là "anh/chị".
Đồng cảm trước — KHÔNG phản bác trực tiếp.
Tái khẳng định giá trị sản phẩm anchor. KHÔNG hạ xuống sản phẩm rẻ hơn.
Nếu khách "để suy nghĩ": đề nghị giữ chỗ hoặc để lại SĐT.
{anchor_rule}
Lịch sử hội thoại: {chat_history}
Ngữ cảnh: {context}""",

    "OUT_OF_SCOPE": """Bạn là nhân viên Akishop.
Xưng "em", gọi khách là "anh/chị".
Câu hỏi nằm ngoài phạm vi. Xin lỗi nhẹ nhàng, KHÔNG bịa.
Hỏi xin SĐT để chuyên viên liên hệ hỗ trợ.""",

    "FOLLOWUP": """Bạn là chuyên viên tư vấn Akishop.
Xưng "em", gọi khách là "anh/chị".
Chỉ dùng thông tin trong Ngữ cảnh. Tiếp tục mạch hội thoại tự nhiên.
{anchor_rule}
Lịch sử hội thoại: {chat_history}
Ngữ cảnh: {context}""",
}

DEFAULT_SYSTEM = """Bạn là chuyên viên tư vấn Akishop.
Xưng "em", gọi khách là "anh/chị".
Chỉ dùng thông tin trong Ngữ cảnh. KHÔNG bịa thông tin.
{anchor_rule}
Lịch sử hội thoại: {chat_history}
Ngữ cảnh: {context}"""

# ══════════════════════════════════════════════════════════════════
# LOAD KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    all_docs, logs = [], []
    for filename, encoding in KNOWLEDGE_FILES.items():
        if not os.path.exists(filename):
            logs.append(("warning", f"Không tìm thấy: {filename}"))
            continue
        try:
            loader = TextLoader(filename, encoding=encoding)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            logs.append(("warning", f"Lỗi {filename}: {e}"))
    if not all_docs:
        return None, None, logs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2800,    # tăng để giữ ngữ cảnh đủ
        chunk_overlap=480,  # overlap lớn hơn để không bị cắt giữa ý
    )
    splits = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # tăng k=5

    llm = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.2,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Akishop Tu Van"},
    )
    return retriever, llm, logs

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def format_chat_history(messages, max_turns=5):
    recent = messages[-(max_turns * 2):]
    return "\n".join(
        ("Khách" if m["role"] == "user" else "Em") + ": " + m["content"]
        for m in recent
    ) or ""

# ══════════════════════════════════════════════════════════════════
# CHAIN LOGIC — INTENT-BASED
# ══════════════════════════════════════════════════════════════════
def build_chain(retriever, llm):
    intent_chain  = INTENT_PROMPT  | llm | StrOutputParser()
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def run(inputs: dict) -> str:
        question = inputs["question"]
        history  = inputs.get("chat_history", "")

        # ── Bước 0: Phát hiện "xin gợi ý" TRƯỚC khi chạy CONDENSE ───
        # Lý do: CONDENSE sẽ hấp thụ câu hỏi kỹ thuật của bot vào câu user
        # (VD: bot hỏi "kích thước + ngân sách?" → user nói "gợi ý đi"
        #  → CONDENSE viết lại thành "Tôi muốn gợi ý với kích thước + ngân sách"
        #  → bot hỏi lại y hệt → vòng lặp vô tận)
        GOI_Y_TRIGGERS = [
            "gợi ý", "không biết", "chưa biết", "chưa có",
            "tư vấn giúp", "mới dùng lần đầu", "không rành",
            "em chưa", "mình chưa", "tôi chưa",
        ]
        is_goi_y = any(t in question.lower() for t in GOI_Y_TRIGGERS)

        # ── Bước 1: Condense TRƯỚC nếu có lịch sử ────────────────
        search_query = question
        if history and not is_goi_y:
            # Trường hợp thường: viết lại câu độc lập từ lịch sử
            search_query = condense_chain.invoke({
                "question": question,
                "chat_history": history,
            }).strip()
        elif history and is_goi_y:
            # FIX LỖI ẢO GIÁC (HALLUCINATION):
            # TUYỆT ĐỐI KHÔNG nhồi biến {history} (chứa định dạng Khách/Em) vào search_query.
            # LLM đã tự đọc được lịch sử từ System Prompt. Chỉ cần truyền lệnh định hướng.
            search_query = f"{question} (Ghi chú cho AI: Khách là người mới, hãy tự đọc nhu cầu trong lịch sử để gợi ý dòng máy phù hợp. TUYỆT ĐỐI KHÔNG tự bịa ra đoạn hội thoại của Khách)."

        # ── Bước 2: Phân loại intent trên câu ĐÃ CONDENSE ────────
        # Bây giờ "Boox Go 7 giá bao nhiêu?" → PRICE (đúng)
        intent = intent_chain.invoke({
            "question": search_query,
            "chat_history": history or "Chưa có",
        }).strip().upper()

        # Chuẩn hoá — fallback nếu model trả về ngoài danh sách
        valid = {"GREETING","PRODUCT","PRICE","COMPARE","POLICY","OBJECTION","OUT_OF_SCOPE","FOLLOWUP"}
        if intent not in valid:
            intent = "PRODUCT"

        # ── Bước 3: Retrieve bằng câu đã condense ─────────────────
        context = ""
        if intent not in ("GREETING", "OUT_OF_SCOPE"):
            docs    = retriever.invoke(search_query)
            context = format_docs(docs)

        # ── Bước 4: Sinh câu trả lời ──────────────────────────────
        system_template = SYSTEM_PROMPTS.get(intent, DEFAULT_SYSTEM)

        # Inject anchor rule + fill các biến
        anchor_rule = ANCHOR_RULE if (history and intent not in ("FOLLOWUP", "GREETING")) else ""
        try:
            system_filled = system_template.format(
                context=context,
                chat_history=history or "Chưa có",
                anchor_rule=anchor_rule,
            )
        except KeyError:
            try:
                system_filled = system_template.format(anchor_rule=anchor_rule)
            except KeyError:
                system_filled = system_template

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_filled),
            ("human", "{question}"),
        ])
        answer_chain = answer_prompt | llm | StrOutputParser()
        # Dùng search_query (đã condense) để AI trả lời đúng ngữ cảnh
        answer = answer_chain.invoke({"question": search_query})

        return answer.strip() if answer and answer.strip() else (
            "Dạ, em xin lỗi, em chưa có thông tin về vấn đề này. "
            "Anh/chị có thể để lại số điện thoại để chuyên viên hỗ trợ ạ."
        )

    return run

# ══════════════════════════════════════════════════════════════════
# CSS — GEMINI STYLE + FIX CON MỌT
# ══════════════════════════════════════════════════════════════════
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background-color: #FFFFFF !important;
    font-family: 'Outfit', sans-serif !important;
    color: #1F1F1F !important;
}

/* ── CON MỌT: góc phải, KHÔNG ăn vào chat input ── */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    bottom: 110px;        /* đẩy lên trên thanh input */
    right: 16px;
    width: 230px;
    height: 230px;
    background-image: url("https://i.postimg.cc/vZSsZM3d/mot.png");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: bottom right;
    pointer-events: none;
    z-index: 1;           /* trên nền, dưới chat bubbles */
    opacity: 0.88;
}

[data-testid="stHeader"], footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    padding-top: 78px !important;
    padding-bottom: 115px !important;
    max-width: 720px !important;
    position: relative;
    z-index: 2;
}

/* ── HEADER ── */
.aki-header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 24px;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid #E8EAED;
}
.aki-header-left { display: flex; align-items: center; gap: 10px; }
.aki-logo { height: 33px; object-fit: contain; }
.aki-header-right { display: flex; align-items: center; gap: 6px; }
.aki-nav-link {
    font-size: 14px; font-weight: 500; color: #5F6368 !important;
    text-decoration: none; padding: 7px 14px; border-radius: 999px; transition: 0.15s;
}
.aki-nav-link:hover { background: #F1F3F4; color: #1F1F1F !important; }
.aki-buy-btn {
    font-size: 13px; font-weight: 600; color: #fff; background: #F07021;
    border: none; border-radius: 999px; padding: 8px 18px; cursor: pointer;
    box-shadow: 0 1px 6px rgba(240,112,33,0.3); transition: 0.15s;
}
.aki-buy-btn:hover { background: #E05A10; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 2px 0 !important;
    display: flex !important; width: 100% !important;
}
[data-testid="stChatMessage"] .stMarkdown p {
    font-size: 15px !important; line-height: 1.7 !important;
    color: #1F1F1F !important; margin: 0 !important;
}

/* USER — phải, pill xám */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatAvatar"] { display: none !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
    background: #F0F4F9 !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 10px 16px !important; max-width: 75% !important;
    margin-left: auto !important; border: none !important; box-shadow: none !important;
}

/* AI — trái, không nền */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    flex-direction: row !important; align-items: flex-start !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 4px 0 !important;
    max-width: 90% !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #F07021, #FF8C42) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(240,112,33,0.3) !important;
    flex-shrink: 0 !important;
}

/* ── CHAT INPUT ── */
[data-testid="stBottom"] {
    background: linear-gradient(to top, #fff 72%, transparent) !important;
}
[data-testid="stBottom"] > div,
.stChatInputContainer, .stChatInputContainer > div { background: transparent !important; }
[data-testid="stChatInput"] > div {
    background: #F0F4F9 !important; border: 1px solid #E8EAED !important;
    border-radius: 24px !important; padding: 10px 18px !important;
    box-shadow: none !important; transition: all 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    background: #fff !important; border-color: #F07021 !important;
    box-shadow: 0 0 0 3px rgba(240,112,33,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #1F1F1F !important;
    caret-color: #F07021 !important; font-size: 15px !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9AA0A6 !important; }
[data-testid="stChatInput"] button { color: #F07021 !important; background: transparent !important; }
[data-testid="stChatInput"] button:hover {
    background: rgba(240,112,33,0.08) !important; border-radius: 50% !important;
}

/* ── NÚT & TEXTAREA ── */
div[data-testid="stButton"] > button {
    background: #F0F4F9 !important; border: 1px solid #E8EAED !important;
    color: #5F6368 !important; border-radius: 10px !important; transition: 0.15s !important;
}
div[data-testid="stButton"] > button:hover { background: #E8EAED !important; color: #1F1F1F !important; }
div[data-testid="stTextArea"] label { display: none !important; }
div[data-testid="stTextArea"] > div > div {
    background: #F0F4F9 !important; border: 1px solid #E8EAED !important; border-radius: 14px !important;
}
div[data-testid="stTextArea"] textarea {
    background: transparent !important; color: #1F1F1F !important;
    font-family: 'Outfit', sans-serif !important;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)
st.markdown("""
<div class="aki-header">
  <div class="aki-header-left">
    <img src="https://i.postimg.cc/PryH01DX/logo.png" class="aki-logo" alt="Akishop">
  </div>
  <div class="aki-header-right">
    <a href="https://akishop.com.vn/" class="aki-nav-link" target="_blank">Sản phẩm</a>
    <a href="https://akishop.com.vn/" class="aki-nav-link" target="_blank">So sánh</a>
    <button class="aki-buy-btn" onclick="window.open('https://akishop.vn','_blank')">🛒 Mua ngay</button>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD ────────────────────────────────────────────────────────────
with st.spinner("Đang tải dữ liệu..."):
    retriever, llm, logs = load_knowledge_base()

for level, msg in (logs or []):
    if level == "warning":
        st.warning(f"⚠️ {msg}")

if retriever is None:
    st.error("Không tìm thấy cam_nang.txt.")
    st.stop()

chain = build_chain(retriever, llm)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "editing_index" not in st.session_state:
    st.session_state.editing_index = None
if "scroll_action" not in st.session_state:
    st.session_state.scroll_action = "idle"  # idle | preserve | bottom

# ── SCROLL MANAGER ──────────────────────────────────────────────────
def inject_scroll_js():
    action = st.session_state.get("scroll_action", "idle")
    js = f"""
    <script>
    (function() {{
        // Tìm container scroll chính của Streamlit
        const getContainer = () =>
            document.querySelector('[data-testid="stAppViewContainer"]') ||
            document.querySelector('section.main') ||
            document.documentElement;

        const action = "{action}";
        const container = getContainer();

        if (action === "preserve") {{
            // Khôi phục vị trí scroll đã lưu
            const saved = sessionStorage.getItem("aki_scroll");
            if (saved !== null) {{
                container.scrollTop = parseInt(saved);
            }}
        }} else if (action === "bottom") {{
            // Cuộn về câu trả lời mới nhất
            setTimeout(() => {{
                container.scrollTo({{ top: container.scrollHeight, behavior: "smooth" }});
            }}, 120);
        }}

        // Liên tục lưu vị trí scroll
        container.addEventListener("scroll", () => {{
            sessionStorage.setItem("aki_scroll", container.scrollTop);
        }}, {{ passive: true }});
    }})();
    </script>
    """
    st.components.v1.html(js, height=0)

inject_scroll_js()
# Reset sau mỗi lần render
st.session_state.scroll_action = "idle"

def run_chain_and_append(question, history_messages):
    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                answer = chain({
                    "question": question,
                    "chat_history": format_chat_history(history_messages),
                })
            except Exception as e:
                answer = f"Lỗi hệ thống: {e}"
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.scroll_action = "bottom"

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            if st.session_state.editing_index == i:
                edited = st.text_area("Sửa:", value=msg["content"], key=f"edit_{i}", height=80)
                c1, c2 = st.columns([1, 5])
                with c1:
                    if st.button("Gửi lại", key=f"sub_{i}"):
                        st.session_state.messages = st.session_state.messages[:i]
                        st.session_state.editing_index = None
                        st.session_state.messages.append({"role": "user", "content": edited})
                        run_chain_and_append(edited, st.session_state.messages[:-1])
                        st.session_state.scroll_action = "bottom"
                        st.rerun()
                with c2:
                    if st.button("Huỷ", key=f"can_{i}"):
                        st.session_state.editing_index = None
                        st.session_state.scroll_action = "preserve"
                        st.rerun()
            else:
                c_msg, c_btn = st.columns([9, 1])
                with c_msg: st.markdown(msg["content"])
                with c_btn:
                    if st.button("✎", key=f"eb_{i}", help="Sửa"):
                        st.session_state.editing_index = i
                        st.session_state.scroll_action = "preserve"
                        st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

if st.session_state.editing_index is None:
    if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        run_chain_and_append(user_input, st.session_state.messages[:-1])
        st.rerun()
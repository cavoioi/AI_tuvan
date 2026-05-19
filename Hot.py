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
INTENT_PROMPT = PromptTemplate.from_template("""Phân tích câu hỏi của khách và phân loại vào MỘT trong các nhóm sau:

ĐỊNH NGHĨA CHÍNH XÁC:
- GREETING: CHỈ khi câu hỏi là chào hỏi thuần túy, KHÔNG đề cập sản phẩm hay nhu cầu cụ thể. Ví dụ: "xin chào", "hello", "shop ơi".
- PRODUCT: hỏi về sản phẩm, tính năng, thông số KỂ CẢ khi có chào hỏi kèm theo. Ví dụ: "xin chào, tư vấn dòng note", "cho tôi hỏi về boox", "máy nào tốt", "đọc và ghi chú", "tôi cần máy để...".
- PRICE: hỏi về giá, khuyến mãi, trả góp, Cam kểt.
- COMPARE: so sánh sản phẩm
- POLICY: hỏi bảo hành, đổi trả, vận chuyển, hỏi về địa chỉ, hệ thống cửa hàng, chi nhánh, giờ làm việc, bảo hành, đổi trả, vận chuyển, thanh toán.
- OBJECTION: phản đối giá, "đắt quá", "để suy nghĩ", do dự
- OUT_OF_SCOPE: hoàn toàn ngoài chủ đề máy đọc sách
- FOLLOWUP: câu trả lời ngắn tiếp nối câu hỏi trước (ví dụ AI hỏi "dùng để làm gì?" khách trả lời "đọc sách")

LƯU Ý QUAN TRỌNG:
- Nếu câu có CHÀO HỎI + YÊU CẦU SẢN PHẨM → PRODUCT (không phải GREETING)
- Nếu câu rất ngắn và có lịch sử hội thoại → FOLLOWUP
- GREETING chỉ dùng khi câu hỏi KHÔNG có thông tin nhu cầu nào

Chỉ trả về đúng một từ khóa, không giải thích.

Lịch sử gần nhất: {chat_history}
Câu hỏi mới: {question}
Loại:""")

# Bước 2: Viết lại câu hỏi độc lập (chỉ dùng khi FOLLOWUP)
CONDENSE_PROMPT = PromptTemplate.from_template("""Khách đang hỏi tiếp theo câu trước. Hãy viết lại câu hỏi thành một câu độc lập, đầy đủ nghĩa.
Không thêm thông tin không có trong câu hỏi mới.

Lịch sử: {chat_history}
Câu hỏi mới: {question}
Câu hỏi độc lập:""")

# ── QUY TẮC ANCHOR (inject vào run() khi build prompt) ───────────
ANCHOR_RULE = """
QUY TẮC TƯ VẤN CÓ ANCHOR SẢN PHẨM (BẮT BUỘC):

BƯỚC 0: NHẬN DIỆN VÀ PHÂN LUỒNG NHU CẦU THEO NHÓM
- Khách thường gọi tắt (Ví dụ: "go 10.3 lumi" = "Boox Go 10.3 Gen 2 Lumi" hoặc khách hỏi go 6 thì = "Boox Go 6", go 7 thì = "Boox Go 7").
+ NẾU đã có đủ thông tin -> Đề xuất đúng máy theo kịch bản + BẮT BUỘC CHÈN KÈM ĐƯỜNG LINK "👉 [Thông tin chi tiết](url)" có trong Ngữ cảnh để khách bấm xem + Kèm theo chiến thuật UP-SALE/CROSS-SELL đã được hướng dẫn.
- KỊCH BẢN PHÂN LOẠI KHI KHÁCH HỎI CHUNG CHUNG HOẶC NÊU NHU CẦU:
  + TRƯỜNG HỢP 0 (HỎI QUÁ CHUNG CHUNG, CHƯA RÕ NHU CẦU): Nếu khách chỉ nói "tư vấn cho tôi dòng boox", "máy nào tốt", "giới thiệu máy đọc sách", "tư vấn cho tôi Kindle"... mà CHƯA CÓ kích thước hay nhu cầu cụ thể. 
    -> TUYỆT ĐỐI KHÔNG liệt kê bất kỳ sản phẩm hay giá tiền nào. BẮT BUỘC phải HỎI LẠI nhu cầu của khách để phân luồng (Ví dụ: "Dạ Boox có rất nhiều dòng máy. Anh/chị đang tìm một chiếc máy nhỏ gọn 6-7 inch để đọc sách chữ, hay cần máy màn hình lớn 10 inch trở lên để ghi chú và đọc PDF ạ?"). 
- TRƯỜNG HỢP 1 (KHÁCH NÊU RÕ MỘT NHU CẦU CỤ THỂ - VD: truyện tranh, sách nói, đọc PDF, học ngoại ngữ...): 
  + BẮT BUỘC tìm mục "KỊCH BẢN TƯ VẤN THEO NHU CẦU ĐỌC CỤ THỂ" trong Ngữ cảnh để đối chiếu.
  + NẾU kịch bản yêu cầu phải hỏi thêm thông tin (VD: hỏi màu hay đen trắng) và Lịch sử CHƯA CÓ -> BẮT BUỘC đặt câu hỏi để làm rõ. (TUYỆT ĐỐI KHÔNG xả sản phẩm vội).
  + NẾU đã có đủ thông tin -> Đề xuất đúng máy theo kịch bản + Kèm theo chiến thuật UP-SALE/CROSS-SELL đã được hướng dẫn.
- TRƯỜNG HỢP 2 (NHU CẦU ĐỌC SÁCH THUẦN TÚY, NHỎ GỌN): Nếu khách tìm máy 6-7 inch, đọc truyện, gọn nhẹ. 
    -> BẮT BUỘC chọn 1 sản phẩm nổi bật nhất trong [NHÓM 1] (Ví dụ: Boox Go Color 7 Gen 2 hoặc Go 7 đen trắng) làm Anchor. Gợi ý thêm: "Nếu anh/chị cần tối giản và tiết kiệm hơn nữa, em có dòng Boox Go 6 hoặc Savi ạ."
- TRƯỜNG HỢP 3 (NHU CẦU GHI CHÚ, LÀM VIỆC): Nếu khách tìm máy màn hình lớn, ghi chú, PDF.
    -> BẮT BUỘC chọn 1 sản phẩm thuộc [NHÓM 2] (Ưu tiên Note Air 5 C) làm Anchor. Gợi ý thêm: "Nếu mình cần xử lý tác vụ nặng, lướt web mượt như tablet, em khuyên mình tham khảo thêm DÒNG TAB SERIES [NHÓM 3] ạ."
  => TUYỆT ĐỐI KHÔNG xả báo giá toàn bộ cửa hàng, chỉ tập trung vào nhóm nhu cầu khách cần.

BƯỚC 1: KIỂM TRA NGÂN SÁCH (NGOẠI LỆ ƯU TIÊN CAO NHẤT)
- Nếu khách đưa ra NGÂN SÁCH CỤ THỂ thấp hơn giá anchor: Áp dụng DOWN-SELL ở Bước 2.
BƯỚC 2: XỬ LÝ THEO ANCHOR & CHIẾN THUẬT DOWN-SELL (KHI KHÁCH CHÊ ĐẮT)
- PHẦN 1: Nếu khách hỏi tính năng -> Xác nhận tính năng trên máy Anchor và trình bày theo BƯỚC 3.
- PHẦN 2 (HẠ CẤP CÙNG PHÂN KHÚC): 
  + Nếu khách chê giá cao hoặc thiếu ngân sách: 
    1. Đưa ra 1 sản phẩm thay thế giá thấp hơn TRONG CÙNG NHÓM NHU CẦU. (Ví dụ: Khách đang ở Nhóm 1 chê Boox Go 7 Color đắt -> Hạ xuống Boox Go 7 đen trắng -> Hạ xuống Go 6 -> Savi. Khách ở Nhóm 2/3 chê Tab đắt -> Hạ xuống Note -> Hạ xuống Go 10.3).
    2. BẮT BUỘC phải giải thích rõ TÍNH NĂNG TƯƠNG TỰ nhưng phải chỉ rõ HẠN CHẾ CỦA MÁY RẺ HƠN (Ví dụ: Xuống Go 7 thì mất màn hình màu, xuống Go 10.3 Gen 2 thì mất đèn nền, xuống Note thì không mượt bằng Tab...).
    3. CUỐI CÙNG, BẮT BUỘC đưa ra CHÍNH SÁCH TRẢ GÓP MẶC ĐỊNH (BƯỚC 4) để chốt sale.

BƯỚC 3: LOGIC TRÌNH BÀY SẢN PHẨM (BẮT BUỘC)
1. Trải nghiệm thực: Đáp ứng nhu cầu như thế nào.
2. Công nghệ: Tính năng nổi bật.
3. Giá & Giải pháp: Đưa ra mức giá chính xác. LUẬT THÉP: BẮT BUỘC phải trích xuất và giữ nguyên 100% đường link "👉 [Thông tin chi tiết](url)" từ Ngữ cảnh và gắn ngay cạnh tên máy hoặc giá tiền. TUYỆT ĐỐI KHÔNG được tự ý bỏ link của sản phẩm. Cuối cùng, dán text trả góp BƯỚC 4 ở cuối (chỉ dán 1 lần).

BƯỚC 4: CHÍNH SÁCH TRẢ GÓP MẶC ĐỊNH (LUẬT THÉP)
TUYỆT ĐỐI KHÔNG dùng từ "trả góp 0%". BẮT BUỘC copy y hệt đoạn sau:
"Akishop có hỗ trợ trả góp online qua thẻ tín dụng, thủ tục trả góp nhanh chóng và tiện lợi. Chi tiết liên hệ Hotline 0856 87 88 89 hoặc qua [Fanpage Akishop](https://www.facebook.com/akishop.official) [Máy đọc sách Akishop](https://akishop.com.vn/) để được tư vấn và hỗ trợ.
• Tại Hà Nội:
- 71 Nguyễn Phong Sắc, Cầu Giấy. Hotline: 0974888717
- 136 Tôn Đức Thắng, Ô Chợ Dừa. Hotline: 0334176893
- 500 Trần Khát Chân, Hai Bà Trưng. Hotline: 0866176500
- Tầng 1 TTTM AEON MALL Hà Đông. Hotline: 0396924602
• Tại Hải Phòng:
- Lô W103 tầng 1, Aeon Mall Hải Phòng, 10 Võ Nguyên Giáp. Hotline: 0844888717
• Tại TP.HCM:
- Akishop Quận 7: 310 Huỳnh Tấn Phát. Hotline: 0336546800
- Akishop Quận 10: 521BIS Cách Mạng Tháng 8. Hotline: 0862726093
- Akishop Bình Thạnh: 2Y Đinh Bộ Lĩnh. Hotline: 0774888717
- Akishop Thủ Đức: 100 Tô Ngọc Vân. Hotline: 0764491367"
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # tăng k=4

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
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

        # ── Bước 1: Condense TRƯỚC nếu có lịch sử ────────────────
        # "giá máy bao nhiêu" + history "boox go 7" → "Boox Go 7 giá bao nhiêu?"
        search_query = question
        if history:
            search_query = condense_chain.invoke({
                "question": question,
                "chat_history": history,
            }).strip()

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
        anchor_rule = ANCHOR_RULE if history else ""
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

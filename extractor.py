import os
import re
import json
from typing import List, Optional, Union, Any
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from utils import get_logger, log_execution_time
import config

logger = get_logger()

# --- Pydantic 모델 정의 (데이터 구조) ---
class SpendingItem(BaseModel):
    """개별 지출 항목"""
    model_config = ConfigDict(populate_by_name=True)

    exec_date: Optional[str] = Field(
        default=None,
        alias="날짜",
        description="해당 항목의 결의일자 또는 실 사용일자를 찾아 YYYY-MM-DD 형식으로 변환. '2025년 07월 11일' 또는 '2025.07.11'과 같이 정확한 날짜가 있으면 반드시 해당 날짜를 사용. (예: 2024-01-15)"
    )
    exec_amount: Optional[int] = Field(
        default=0,
        alias="금액",
        description="해당 항목의 집행 금액. 표에 '금회 집행금액'과 '누계 사용 금액'이 모두 존재할 경우, 반드시 '누계 사용 금액'을 추출. 쉼표/통화기호 제거 후 숫자(Integer) 형태로 반환."
    )
    exec_spender: Optional[str] = Field(
        default=None,
        alias="성명",
        description="집행을 담당한 사람의 '성명'. 결재란이나 담당자란에서 1순위 '안전관리자', 2순위 '현장대리인' 성명을 추출. 직책(안전관리자, 현장대리인, 사원 등)은 제거하고 이름만 반환. (공백 모두 제거 필수). 날짜(예: 2024-01-01)나 금액은 절대로 포함하지 마십시오."
    )
    exec_item: Optional[str] = Field(
        default=None,
        alias="내역",
        description="문서 상의 구체적인 사용 내역 항목 명칭."
    )
    std_item: Optional[str] = Field(
        default=None,
        alias="표준항목",
        description="문서 상의 구체적인 사용 내역 항목 명칭을 참고하여 표준 항목 리스트 중 하나로 매핑."
    )

    @field_validator('exec_date', mode='before')
    @classmethod
    def parse_date(cls, v):
        """
        다양한 날짜 형식을 YYYY-MM-DD로 정규화.
        변환 불가 시 None 반환 (잘못된 형식 방지)
        """
        if not v:
            return None
        v_str = str(v).strip()

        # 빈 문자열이나 무의미한 값 체크
        if not v_str or v_str in ["-", "N/A", "null", "None", ""]:
            return None

        # 이미 YYYY-MM-DD 형식이면 그대로 반환
        if re.match(r'^\d{4}-\d{2}-\d{2}$', v_str):
            return v_str

        try:
            # 다양한 패턴에서 숫자 추출
            # "2025년 04월 11일", "2025.04.11", "2025/04/11", "25-04-11" 등
            nums = re.findall(r'\d+', v_str)

            if len(nums) >= 3:
                # 년, 월, 일이 모두 있는 경우
                year, month, day = int(nums[0]), int(nums[1]), int(nums[2])
                if year < 100:
                    year += 2000
                # 유효성 검사
                if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year:04d}-{month:02d}-{day:02d}"

            elif len(nums) == 2:
                # 년, 월만 있는 경우 → 해당 월 1일로 설정
                year, month = int(nums[0]), int(nums[1])
                if year < 100:
                    year += 2000
                if 1900 <= year <= 2100 and 1 <= month <= 12:
                    return f"{year:04d}-{month:02d}-01"

            # YYYY-MM-DD 패턴 직접 검색
            match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', v_str)
            if match:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                return f"{year:04d}-{month:02d}-{day:02d}"

            # 변환 실패 시 None 반환 (잘못된 형식 방지)
            return None

        except Exception:
            return None

    @field_validator('exec_amount', mode='before')
    @classmethod
    def parse_amount(cls, v):
        if v is None: return 0
        if isinstance(v, int): return v
        clean_str = str(v).replace(",", "").replace("원", "").replace("O", "0").replace("o", "0").strip()
        if not clean_str.isdigit():
            nums = re.findall(r'\d+', clean_str)
            clean_str = "".join(nums) if nums else "0"
        try: return int(clean_str)
        except: return 0

    @field_validator('exec_spender', mode='before')
    @classmethod
    def clean_spender(cls, v):
        if not v: return None
        v = str(v).strip()
        if re.search(r'\d{4}[-./년]\d{1,2}', v) or re.search(r'\d+원', v): return None
        for term in config.REMOVE_TERMS: v = v.replace(term, "")
        # 정규식 패턴 완화: 이름이 포함된 문자열에서 한글만 추출 후 길이를 더 유연하게 체크
        korean = "".join(re.findall(r'[가-힣]+', v))
        if 2 <= len(korean) <= 5: # 외자나 4~5글자 이름 고려 (OCR 노이즈 포함 대비)
            return korean
        return None

class SpendingList(BaseModel):
    """지출 내역 리스트 - 다양한 LLM 출력 형태를 유연하게 처리"""
    items: List[SpendingItem] = Field(default_factory=list, description="추출된 지출 내역 리스트")

    @model_validator(mode='before')
    @classmethod
    def normalize_llm_output(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. 'items' 키가 직접 존재하는 경우
        if "items" in data and isinstance(data["items"], list):
            return data

        # 2. '항목별 사용내역' (phi4 스타일) 확인
        if "항목별 사용내역" in data:
            return cls._convert_phi4_format(data)

        # 3. [재귀적 탐색] 모든 키를 뒤져서 가장 긴 리스트를 가진 항목을 items로 채택
        # LLM이 제목을 키로 썼을 때 ('== 2025년... ==') 이를 해결하는 핵심 로직
        best_list = []
        
        def find_lists(obj):
            nonlocal best_list
            if isinstance(obj, list):
                if len(obj) > len(best_list):
                    best_list = obj
            elif isinstance(obj, dict):
                for v in obj.values():
                    find_lists(v)

        find_lists(data)
        
        if best_list:
            logger.info(f"Successfully found list with {len(best_list)} items in nested structure.")
            return {"items": best_list}

        # 4. 단일 객체인 경우
        if any(k in data for k in ["exec_date", "날짜", "exec_amount", "금액"]):
            return {"items": [data]}

        logger.warning(f"Failed to find any list in LLM output. Keys: {list(data.keys())}")
        return {"items": []}

    @classmethod
    def _convert_phi4_format(cls, data: dict) -> dict:
        items = []
        usage_data = data.get("항목별 사용내역", {})
        writers = data.get("작성자", [])
        spender = None
        for writer in (writers if isinstance(writers, list) else []):
            if "안전관리자" in writer.get("직책", ""):
                spender = writer.get("성명")
                break
        if isinstance(usage_data, dict):
            for name, details in usage_data.items():
                if not isinstance(details, dict): continue
                amount = details.get("누계사용 금액") or details.get("금회 집행 금액") or details.get("금액")
                if amount:
                    items.append({
                        "exec_item": name, "exec_amount": amount,
                        "exec_spender": spender, "exec_date": data.get("작성일")
                    })
        return {"items": items}

# --- 핵심 추출 함수 ---
@log_execution_time(logger)
def extract_data_with_llm(
    content: str,
    llm_type: str = "openai",
    model_name: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    top_p: float = 1.0
) -> List[dict]:

    """
    LangChain PydanticOutputParser와 ChatModel(OpenAI or Ollama)을 사용하여
    제공된 텍스트 콘텐츠에서 구조화된 지출 데이터를 추출합니다.
    """
    # 모델명 기본값 설정
    if model_name is None:
        model_name = config.OPENAI_MODEL_DEFAULT if llm_type == "openai" else config.OLLAMA_MODEL_DEFAULT


    # 1. Output Parser 설정
    parser = PydanticOutputParser(pydantic_object=SpendingList)

    # 2. Prompt 템플릿 설정
    system_prompt = """
    당신은 안전관리비 사용 내역서 집행 내역 추출 자동화 처리 전문가입니다.
    제공된 텍스트 데이터의 표(Table)를 분석하여, **각 행(Row)마다 개별적인 집행 내역**을 추출하십시오.

    [표준 항목 리스트]
    {standard_items}

    [핵심 추출 가이드]
    1. **개별 추출 (One Item Per Row)**:
       - 표의 각 행(Row)은 하나의 독립적인 집행 내역입니다. **절대로 여러 행을 하나로 합치지 마십시오.**
       - 예를 들어, 표에 3개의 유효한 행이 있다면, 결과 리스트(items)에는 3개의 객체가 있어야 합니다.

    2. **날짜(Date) 처리**:
       - **1순위 (행 내부)**: 먼저 해당 행(Row)에 날짜가 있는지 확인하고 있다면 그 날짜를 사용하십시오.
       - **2순위 (문서 전체)**: 행에 날짜가 없다면, **문서 하단(서명란)**이나 **상단(제목)** 등에 있는 **'작성일', '신청일' 또는 독립적인 날짜(예: ## 2025년 07월 11일)**를 찾아 공통적으로 적용하십시오.
       - **주의**: 문서 내에 '11일', '25일' 등 구체적인 일자(Day)가 명시되어 있다면, **절대로 '01'일로 임의 변환하지 말고 해당 일자를 그대로(예: 2025-07-11) 사용하십시오.** (가장 중요)
       - 구체적인 일(Day) 정보가 문서 전체를 찾아봐도 **전혀 없는 경우에만** 최후의 수단으로 '01'일을 사용하십시오.

    3. **행(Row) 단위 매핑**:
       - 반드시 해당 행(Same Row)에 있는 항목명과 금액만 매핑하십시오. 다른 행의 정보를 섞지 마십시오.

    4. **금액 추출 (Strict Rules)**:
       - 표의 여러 금액 컬럼 중 **'가장 최종적인 누적 금액(Total/Accumulated)'**을 추출하십시오.
       - '금회 집행(Current)'이 아닌 '전월 누계 + 금회'인 **'누계(Accumulated)'** 금액이 기준입니다.
       - 단, '누계' 컬럼이 없고 '금회'만 있는 경우 '금회'를 사용하십시오. 상황에 맞춰 가장 큰 의미의 집행 금액을 선택하십시오.
       - 금액에 포함된 '원', ','(쉼표) 등은 모두 제거하고 순수 숫자(Integer)만 반환해야 합니다.

    5. **제외 대상**:
       - **'소계', '합계', 'Total', '월계', '총계', '총 계' 행은 절대 추출하지 마십시오.** (매우 중요)
       - 금액이 0원인 항목은 추출하지 마십시오. '-'(하이픈)이나 비어있는 경우도 0원으로 간주하여 제외합니다.

    6. **집행자 (Spender) 추출 규칙**:
       - 결재란/담당자란에서 **1순위: '안전관리자'**, **2순위: '현장대리인'** 성명을 찾아 사용하십시오.
       - 안전관리자가 없으면 현장대리인 성명을 사용하십시오.
       - 행 내에 특정 사용자 이름이 있다면 그것을 사용하십시오.

    [JSON 포맷팅 엄격 준수] (가장 중요)
    - 반드시 아래 JSON 스키마를 정확히 따르는 **단일 JSON 객체**를 반환하십시오.
    - `items` 리스트 안에 **모든 유효한 행**을 각각 별도의 객체로 추가해야 합니다.
    - **절대로 '== 제목 ==', 인사말, 설명 또는 Markdown 코드 블록(```json)을 포함하지 마십시오.**
    - 응답의 시작은 반드시 중괄호 열기로 시작하고, 중괄호 닫기로 끝나야 합니다.
    - Markdown 태그(```json)도 사용하지 마십시오.
    - JSON 외의 어떤 텍스트도 출력하지 마십시오.

    JSON Schema:
    {{
      "items": [
        {{
          "exec_date": "YYYY-MM-DD",
          "exec_amount": 9020000,
          "exec_spender": "이름",
          "exec_item": "안전시설비 등",
          "std_item": "안전 시설비 등"
        }},
        {{
          "exec_date": "YYYY-MM-DD",
          "exec_amount": 20475000,
          "exec_spender": "이름",
          "exec_item": "안전보호구 및 안전 장구 구입비 등",
          "std_item": "보호구 등"
        }}
        // ... (나머지 행들)
      ]
    }}
    """

    import time
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Timestamp: {current_time}\nDocument Content:\n\n{content}")
    ])

    # 3. Model 설정
    if llm_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set.")
            raise EnvironmentError("OPENAI_API_KEY environment variable not found.")

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            api_key=api_key
        )
    elif llm_type == "ollama":
        logger.info(f"Connecting to Local LLM (Ollama) at {config.OLLAMA_HOST} with model {model_name}")
        llm = ChatOllama(
            base_url=config.OLLAMA_HOST,
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            format="json",
            num_ctx=16384,      # 컨텍스트 윈도우 확장 (기본값 2048 → 16384)
            num_predict=4096,   # 출력 토큰 확장 (기본값 128 → 4096)
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}. Use 'openai' or 'ollama'.")

    # 4. Robust JSON Parser: 텍스트 내에서 JSON만 강제로 추출
    def robust_chain_invoke(input_data):
        raw_res = (prompt | llm).invoke(input_data)
        text = raw_res.content if hasattr(raw_res, 'content') else str(raw_res)

        # [DEBUG] LLM raw 응답 출력
        logger.info(f"[DEBUG] LLM raw response (first 500 chars):\n{text[:500]}")
        logger.info(f"[DEBUG] LLM raw response (last 500 chars):\n{text[-500:]}")

        # 1. ```json ... ``` 블록 찾기
        json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_block:
            text = json_block.group(1)
            logger.info("[DEBUG] Extracted JSON from code block")
        else:
            # 2. 가장 바깥쪽 { } 찾기
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
                logger.info(f"[DEBUG] Extracted JSON via regex, length: {len(text)}")

        return parser.parse(text)

    logger.info(f"Calling LLM ({llm_type}:{model_name}) via LangChain for data extraction...")

    try:
        result = None
        input_data = {
            "content": content, 
            "standard_items": config.STANDARD_ITEMS,
            "current_time": str(time.time())
        }

        if llm_type == "openai":
            # Cost Tracking with Callback (OpenAI only)
            with get_openai_callback() as cb:
                result = robust_chain_invoke(input_data)

                # 비용 계산 및 로그
                in_t = cb.prompt_tokens
                out_t = cb.completion_tokens
                total_t = cb.total_tokens

                price_info = config.PRICING.get(model_name)
                if price_info:
                    cost_usd = (in_t / 1_000_000 * price_info["input"]) + \
                               (out_t / 1_000_000 * price_info["output"])
                    cost_krw = cost_usd * config.EXCHANGE_RATE

                    logger.info(f"[{model_name}] Token Usage: In={in_t}, Out={out_t}, Total={total_t}")
                    logger.info(f"[{model_name}] Est. Cost: ${cost_usd:.6f} ({cost_krw:.2f} KRW)")
                else:
                    logger.info(f"Token Usage provided by callback: {cb}")

        else:
            # Ollama execution (No cost tracking)
            result = robust_chain_invoke(input_data)
            logger.info(f"Local LLM ({model_name}) extraction completed.")

        # 결과 처리
        extracted_items = []
        if result and result.items:
            for item in result.items:
                # 0원 제외 정책
                if item.exec_amount == 0:
                    continue

                # 객체 덤프
                item_dict = item.model_dump()
                extracted_items.append(item_dict)

        if not extracted_items:
            logger.warning("No items extracted. Check if the content contains valid table data.")
            logger.warning(f"Content snippet sent to LLM:\n{content[:1000]}...")

        logger.info(f"Extraction successful. Extracted {len(extracted_items)} items.")

        return extracted_items


    except Exception as e:
        logger.error(f"LLM API call or parsing failed: {e}")
        raise e

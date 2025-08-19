"""
LLM 체인 설정
"""
from typing import Dict
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

import params
from prompt import talking_prompt

# 인메모리 저장소
in_memory_store: Dict[str, ChatMessageHistory] = {}

# LLM 모델
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=params.OPENAI_API_KEY,
    temperature=0.7,
    streaming=False
)

def get_message_history(session_id: str):
    """세션 히스토리 가져오기"""
    if params.USE_REDIS:
        # Redis 사용 - TTL 자동 관리
        return RedisChatMessageHistory(
            session_id, 
            url=params.REDIS_URL, 
            ttl=3600
        )
    else:
        # 인메모리 사용
        if session_id not in in_memory_store:
            in_memory_store[session_id] = ChatMessageHistory()
        return in_memory_store[session_id]

def talking_chain():
    """대화 체인 생성"""
    chain = (
        {
            "question": itemgetter("query"),
            "chat_history": itemgetter("chat_history"),
        }
        | talking_prompt 
        | llm
        | StrOutputParser()
    )
    
    return RunnableWithMessageHistory(
        chain,
        get_message_history,
        input_messages_key="query",
        history_messages_key="chat_history",
    )

def clear_memory_sessions():
    """인메모리 세션 정리 - 50개 넘으면 오래된 것부터 삭제"""
    global in_memory_store
    
    if len(in_memory_store) > 50:
        # 최신 25개만 유지
        keep_sessions = list(in_memory_store.keys())[-25:]
        in_memory_store = {k: in_memory_store[k] for k in keep_sessions}
        print(f"Session cleanup: keeping {len(in_memory_store)} recent sessions")

import asyncio
import json
import hashlib

import httpx
from openai import AsyncOpenAI

from app import database as db
from app.embedding import encode_text
from app.tools import execute_tool, TOOLS_SCHEMA
from config import settings

FALLBACK_PROVIDERS = [
    {
        "name": "mimo",
        "api_key": settings.mimo_api_key,
        "base_url": settings.mimo_base_url,
        "model": settings.mimo_model,
    },
    {
        "name": "qwen",
        "api_key": settings.qwen_api_key,
        "base_url": settings.qwen_base_url,
        "model": settings.qwen_model,
    },
    {
        "name": "deepseek",
        "api_key": settings.deepseek_api_key,
        "base_url": settings.deepseek_base_url,
        "model": settings.deepseek_model,
    },
]


_clients = {}

def _get_client(api_key: str, base_url: str) -> AsyncOpenAI:
    client_key = f"{api_key}_{base_url}"
    if client_key not in _clients:
        _clients[client_key] = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(timeout=60.0),
        )
    return _clients[client_key]

async def _chat_with_fallback(messages: list, tools: list = None, tool_choice: str = "auto", temperature: float = 0.4, report_callback=None):
    last_error = None
    for provider in FALLBACK_PROVIDERS:
        try:
            if report_callback:
                await report_callback("analyzing", f"正在尝试使用 {provider['name']} 模型思考...")
            
            client = _get_client(provider["api_key"], provider["base_url"])
            response = await client.chat.completions.create(
                model=provider["model"],
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
            )
            return response
        except Exception as e:
            print(f"[agent] Provider {provider['name']} failed: {e}")
            last_error = e
            continue
    raise Exception(f"All fallback models failed. Last error: {last_error}")

def _get_query_hash(query: str, game_name: str) -> str:
    return hashlib.md5(f"{game_name}:{query}".encode("utf-8")).hexdigest()

def _get_default_client() -> AsyncOpenAI:

    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            http_client=httpx.AsyncClient(timeout=60.0),
        )
    return _client


async def _llm_chat(prompt: str, max_tokens: int = 2000, temperature: float = 0.4) -> str:
    """Call the LLM with retry. Returns empty string on persistent failure."""
    client = _get_default_client()
    last_error = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            print(f"[rag] LLM returned empty content (attempt {attempt + 1})")
        except Exception as e:
            last_error = e
            print(f"[rag] LLM call failed (attempt {attempt + 1}): {e}")
        await asyncio.sleep(1.5)
    if last_error:
        print(f"[rag] LLM persistent failure: {last_error}")
    return ""


SYSTEM_PROMPT = """浣犳槸涓€涓笓涓氱殑娓告垙淇℃伅闂瓟鍔╂墜銆傝鍩轰簬鎻愪緵鐨勫弬鑰冧笂涓嬫枃鍥炵瓟鐢ㄦ埛闂銆?
瑕佹眰:
- 缁煎悎鍙傝€冭祫鏂欎腑鐨勪俊鎭?缁欏嚭璇︾粏銆佸噯纭€佺粨鏋勫寲鐨勫洖绛?- 寮曠敤鏉ユ簮鏃朵娇鐢?[1], [2] 绛夋爣璁?- 鍙互瀵规绱㈠埌鐨勯浂鏁ｄ俊鎭繘琛屾暣鍚堛€佽ˉ鍏ㄣ€佷紭鍖?浣垮洖绛旀祦鐣呭畬鏁?- 濡傛灉鍙傝€冭祫鏂欑‘瀹炰笉鍖呭惈鐩稿叧闂绛旀,鍥炲:"褰撳墠鏁版嵁搴撶煡璇嗕笉瓒?鏈兘妫€绱㈠埌浣犳彁闂殑淇℃伅銆?
- 涓嶈缂栭€犺祫鏂欎腑涓嶅瓨鍦ㄧ殑淇℃伅,鍥炵瓟浣跨敤涓枃"""

KNOWLEDGE_INSUFFICIENT = "褰撳墠鏁版嵁搴撶煡璇嗕笉瓒筹紝鏈兘妫€绱㈠埌浣犳彁闂殑淇℃伅銆?

# 绠＄嚎鐗规€ц嚜鎻忚堪 - 璇勬祴鏃跺啓鍏?config_snapshot,渚夸簬鍖哄垎涓嶅悓鐗堟湰绠＄嚎鐨勬寚鏍囥€?# 鏂板/鍏抽棴鏌愰」妫€绱紭鍖栨椂璇峰悓姝ヤ慨鏀规澶?璁?Excel 閲岀殑鐗堟湰鏍囪鍑嗙‘鍙嶆槧绠＄嚎鐘舵€併€?PIPELINE_FEATURES = {
    "hyde": True,               # 鍋囪鎬у洖绛旇緟鍔╂绱?(Step 1)
    "context_window": True,     # 鍙洖鍛戒腑娈佃惤鍙婂叾鍓嶅悗鐩搁偦娈佃惤 (retrieve_with_fallback)
    "dynamic_threshold": True,  # 鏍囧噯闃堝€兼棤缁撴灉鏃堕檷浣庨槇鍊奸噸璇?}


# 鈹€鈹€ Step 1: 鐢熸垚鍋囪鎬у洖绛?(HyDE) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

async def generate_hypothetical_answer(game_name: str, user_message: str) -> str:
    """LLM 鍒嗘瀽闂,鐢熸垚鍋囪鎬у洖绛斻€?
    鍋囪鎬у洖绛旀槸闄堣堪鍙?璇箟绌洪棿鏇存帴杩戠煡璇嗗簱涓殑鐧剧鏂囨。,
    鐢ㄥ畠妫€绱㈡瘮鐢ㄥ師濮嬮棶棰樻绱㈠彫鍥炵巼鏇撮珮(HyDE 鍘熺悊)銆?    鍗充娇鍥炵瓟涓嶅噯纭篃鏃犲Θ,鍙渶鍖呭惈鐩稿叧鏈鍜屾蹇点€?    """
    prompt = f"""鐢ㄦ埛鎯充簡瑙ｆ父鎴忋€妠game_name}銆嬬殑浠ヤ笅闂:
{user_message}

璇峰熀浜庝綘瀵硅繖娆炬父鎴忕殑浜嗚В,鐢熸垚涓€涓畝鐭殑鍋囪鎬у洖绛?150-300瀛?銆?瑕佹眰:
1. 鍗充娇涓嶇‘瀹氫篃瑕佺粰鍑哄彲鑳界殑绛旀,缁濆涓嶈鎷掔粷鎴栬"鎴戜笉鐭ラ亾"
2. 鍖呭惈鐩稿叧鐨勬父鎴忔湳璇€佹蹇点€佹満鍒跺悕绉般€佹暟鍊笺€佸湴鐐广€佽鑹插悕
3. 鐢ㄩ檲杩板彞,鍍忕櫨绉戠煡璇嗘潯鐩竴鏍峰啓浣?4. 杩欎釜鍥炵瓟浠呯敤浜庣煡璇嗗簱妫€绱㈣緟鍔?涓嶉渶瑕佸畬鍏ㄥ噯纭?
鍋囪鎬у洖绛?"""
    return await _llm_chat(prompt, max_tokens=500, temperature=0.5)


async def extract_game_name(user_message: str, history_msgs: list[dict]) -> str:
    """LLM 浠庣敤鎴疯緭鍏ュ拰鍘嗗彶璁板綍涓彁鍙栨父鎴忓悕绉般€?""
    prompt = f"""璇蜂粠鐢ㄦ埛鐨勬彁闂拰涓婁笅鏂囧巻鍙蹭腑锛屾彁鍙栧嚭鐢ㄦ埛褰撳墠璁ㄨ鐨勫叿浣撴父鎴忓悕绉般€?濡傛灉鏄庣‘鎻愬埌浜嗘煇涓父鎴忓悕绉帮紝鎴栬€呮牴鎹笂涓嬫枃鑳芥槑纭垽鏂槸鍝娓告垙锛岃浠呬粎杈撳嚭璇ユ父鎴忓悕绉帮紙渚嬪锛氬灏旇揪浼犺鏃烽噹涔嬫伅銆佸師绁烇級锛屼笉瑕佸寘鍚换浣曞浣欑殑瀛楄瘝銆佹爣鐐规垨瑙ｉ噴銆?濡傛灉娌℃湁鎻愬埌浠讳綍鍏蜂綋鐨勬父鎴忥紝鎴栬€呮棤娉曠‘瀹氾紝璇风洿鎺ヨ緭鍑?"None"銆?
鐢ㄦ埛鎻愰棶: {user_message}"""
    if history_msgs:
        history_text = "\n".join([f"{'鐢ㄦ埛' if m['role']=='user' else '鍔╂墜'}: {m['content'][:200]}" for m in history_msgs[-3:]])
        prompt += f"\n\n杩戞湡鍘嗗彶璁板綍:\n{history_text}"
        
    extracted = await _llm_chat(prompt, max_tokens=20, temperature=0.0)
    extracted = extracted.strip()
    if not extracted or extracted.lower() in ("none", "null", "涓嶇煡閬?, "鏈寚瀹?):
        return ""
    return extracted


# 鈹€鈹€ Step 2: 娣峰悎妫€绱?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def merge_and_dedupe(result_lists: list[list[dict]], top_k: int) -> list[dict]:
    """鍚堝苟澶氳矾妫€绱㈢粨鏋?鎸夋枃妗?id 鍘婚噸,鍙栨瘡绡囨渶楂樼浉浼煎害,鎺掑簭鍚庡彇 top_k銆?""
    seen: dict[int, dict] = {}
    for results in result_lists:
        for r in results:
            doc_id = r["id"]
            if doc_id not in seen or r["similarity"] > seen[doc_id]["similarity"]:
                seen[doc_id] = r
    merged = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)
    return merged[:top_k]


def retrieve_with_fallback(query_vec, game_name: str, top_k: int) -> list[dict]:
    """
    璇箟浼樺厛 + 涓婁笅鏂囩獥鍙ｆ墿鍏呮绱€?    1. 鍏堝湪 semantic_chunks 琛ㄤ腑鎼滅储璇箟鏈€鐩歌繎鐨勫彞瀛愩€?    2. 濡傛灉浠?settings.similarity_threshold 涓洪槇鍊兼棤缁撴灉锛屽垯闄嶄綆闃堝€奸噸璇曘€?    3. 鑾峰彇鍖归厤鍙ュ瓙鎵€褰掑睘鐨?parent chunk 鍙婂叾鍓嶅悗鐨勭浉閭?chunk (Context Window)銆?    4. 鍚堝苟鐩搁偦 chunk 鐨勬枃鏈紝鍘婚噸锛岀粍瑁呮垚瀹屾暣涓斾笂涓嬫枃杩炶疮鐨勫€欓€夋绱㈢粨鏋溿€?    """
    threshold = settings.similarity_threshold
    semantic_matches = db.search_similar_semantic(
        query_vec, game_name, top_k=top_k, threshold=threshold
    )

    if not semantic_matches:
        lowered = max(0.1, threshold - 0.15)
        print(f"[rag] Semantic search: Standard threshold {threshold} returned nothing, retrying with {lowered}")
        semantic_matches = db.search_similar_semantic(
            query_vec, game_name, top_k=top_k, threshold=lowered
        )

    if not semantic_matches:
        return []

    # 鑱氬悎鐖舵枃妗ｅ苟鎷夊彇涓婁笅鏂囩獥鍙?    seen_parent_ids = set()
    results = []

    for match in semantic_matches:
        doc_id = match["document_id"]
        if doc_id in seen_parent_ids:
            continue
        seen_parent_ids.add(doc_id)

        # 鍙洖璇ユ钀藉強鍏跺墠鍚庣浉閭荤殑娈佃惤
        context_docs = db.get_document_with_context(doc_id)
        if not context_docs:
            continue

        # 鎷兼帴娈佃惤鍐呭
        combined_content = "\n\n".join([doc["content"] for doc in context_docs])

        # 鎵惧埌琚懡涓殑涓绘钀?        main_doc = next((d for d in context_docs if d["id"] == doc_id), context_docs[0])

        # 鏋勯€犲彫鍥炴潯鐩紝骞跺姞涓婃爣绛?(tag) 鎻愪緵缁?LLM
        tag_str = f"銆愪富棰橈細{match['tag']}銆慭n" if match["tag"] else ""
        results.append({
            "id": main_doc["id"],
            "content": tag_str + combined_content,
            "title": main_doc["title"],
            "url": main_doc["url"],
            "source_name": main_doc["source_name"],
            "similarity": match["similarity"],
            "chunk_index": main_doc["chunk_index"],
        })

    # 鎸夌浉浼煎害閲嶆柊鎺掑簭
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# 鈹€鈹€ Step 1-2 灏佽: HyDE + 娣峰悎妫€绱?(渚?rag_query 涓庤瘎娴嬪叡鐢? 鈹€鈹€鈹€鈹€鈹€鈹€

async def retrieve_documents(
    game_name: str,
    user_message: str,
    progress_callback=None,
    verbose: bool = True,
) -> list[dict]:
    """HyDE + 娣峰悎妫€绱?+ 鍚堝苟鍘婚噸銆?
    灏佽鍘?rag_query 鐨?Step 1-2(鍋囪鍥炵瓟鐢熸垚 + 鍘熷闂/鍋囪鍥炵瓟鍙岃矾妫€绱?+ 鍘婚噸),
    **鏃?DB 鍐欏壇浣滅敤**,鏃㈣ rag_query 璋冪敤,涔熻 eval 璇勬祴璋冪敤,
    淇濊瘉璇勬祴涓庣敓浜ф绱㈤€昏緫鍚屾簮銆佷笉浼氶殢杩唬鑰屽垎鍙夈€?
    Args:
        progress_callback: 鍙€?async callable(stage: str, message: str[, content])銆?            瀛樺湪鏃跺湪姣忎釜闃舵璋冪敤,鐢ㄤ簬鎺ㄩ€佸疄鏃惰繘搴?rag_query 浼犲叆鍏?_report 闂寘)銆?        verbose: 鏄惁鎵撳嵃闃舵鏃ュ織銆傜敓浜ц矾寰勪繚鎸?True;璇勬祴鎵归噺璺戞椂浼?False 鍑忓皯鍣０銆?
    Returns: 鎸夌浉浼煎害闄嶅簭鎺掑垪鐨勬绱㈢粨鏋滃垪琛?姣忎釜鍏冪礌鍚?id/content/similarity/title/url 绛夈€?    """
    async def _report(stage: str, message: str) -> None:
        if progress_callback is not None:
            try:
                await progress_callback(stage, message)
            except Exception:
                # 鍥炶皟澶辫触涓嶅簲褰卞搷妫€绱富娴佺▼
                pass

    # Step 1: LLM 鍒嗘瀽闂,鐢熸垚鍋囪鎬у洖绛?(HyDE)
    await _report("analyzing", "姝ｅ湪鍒嗘瀽闂锛屾瀯鎬濆亣璁惧洖绛斺€?)
    if verbose:
        print(f"[rag] Step 1: Generating hypothetical answer for 銆妠game_name}銆?..")
    hypothetical = await generate_hypothetical_answer(game_name, user_message)
    if verbose:
        if hypothetical:
            print(f"[rag] Hypothetical answer: {hypothetical[:80]}...")
        else:
            print("[rag] Hypothetical answer generation failed, falling back to query-only retrieval")

    # Step 2: 娣峰悎妫€绱?鍋囪鍥炵瓟 + 鍘熷闂)
    await _report("retrieving", "姝ｅ湪妫€绱㈢煡璇嗗簱锛屽尮閰嶇浉鍏冲唴瀹光€?)
    if verbose:
        print("[rag] Step 2: Hybrid retrieval (HyDE + original query)...")
    result_lists = []

    # 2a. 鍘熷闂妫€绱?    query_vec = encode_text(user_message)
    result_lists.append(retrieve_with_fallback(query_vec, game_name, settings.top_k))

    # 2b. 鍋囪鍥炵瓟妫€绱?濡傛灉鐢熸垚鎴愬姛)
    if hypothetical:
        hyde_vec = encode_text(hypothetical)
        result_lists.append(retrieve_with_fallback(hyde_vec, game_name, settings.top_k))

    # 鍚堝苟鍘婚噸
    retrieved = merge_and_dedupe(result_lists, settings.top_k)
    if verbose:
        print(f"[rag] Retrieved {len(retrieved)} unique documents after merge")
    return retrieved


# 鈹€鈹€ Step 5: LLM 浼樺寲鐢熸垚鏈€缁堝洖绛?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

async def generate_final_answer(
    user_message: str,
    retrieved: list[dict],
    history_msgs: list[dict],
) -> str:
    """LLM 鍩轰簬妫€绱㈢粨鏋滀紭鍖栫敓鎴愭渶缁堝洖绛斻€?""
    # 鏋勫缓涓婁笅鏂?    context_parts = []
    for i, doc in enumerate(retrieved):
        title = doc.get("title") or "鏈懡鍚?
        url = doc.get("url") or ""
        context_parts.append(f"[鏉ユ簮{i+1}: {title} ({url})]\n{doc['content']}")

    # 鏋勫缓鍘嗗彶
    history_parts = []
    for msg in history_msgs:
        role_label = "鐢ㄦ埛" if msg["role"] == "user" else "鍔╂墜"
        history_parts.append(f"{role_label}: {msg['content']}")

    # 缁勮 prompt
    full_prompt = SYSTEM_PROMPT
    if context_parts:
        full_prompt += "\n\n鍙傝€冧笂涓嬫枃:\n" + "\n\n".join(context_parts)
    if history_parts:
        full_prompt += "\n\n瀵硅瘽鍘嗗彶:\n" + "\n".join(history_parts)
    full_prompt += f"\n\n鐢ㄦ埛闂: {user_message}\n鍥炵瓟(涓枃,寮曠敤鏉ユ簮璇锋爣娉ㄧ紪鍙?:"

    answer = await _llm_chat(full_prompt, max_tokens=2000, temperature=0.3)
    return answer if answer else KNOWLEDGE_INSUFFICIENT


async def generate_final_answer_stream(
    user_message: str,
    retrieved: list[dict],
    history_msgs: list[dict],
    on_token_callback=None,
) -> str:
    """LLM 鍩轰簬妫€绱㈢粨鏋滀紭鍖栫敓鎴愭渶缁堝洖绛旓紝鏀寔娴佸紡杈撳嚭銆?""
    # 鏋勫缓涓婁笅鏂?    context_parts = []
    for i, doc in enumerate(retrieved):
        title = doc.get("title") or "鏈懡鍚?
        url = doc.get("url") or ""
        context_parts.append(f"[鏉ユ簮{i+1}: {title} ({url})]\n{doc['content']}")

    # 鏋勫缓鍘嗗彶
    history_parts = []
    for msg in history_msgs:
        role_label = "鐢ㄦ埛" if msg["role"] == "user" else "鍔╂墜"
        history_parts.append(f"{role_label}: {msg['content']}")

    # 缁勮 prompt
    full_prompt = SYSTEM_PROMPT
    if context_parts:
        full_prompt += "\n\n鍙傝€冧笂涓嬫枃:\n" + "\n\n".join(context_parts)
    if history_parts:
        full_prompt += "\n\n瀵硅瘽鍘嗗彶:\n" + "\n".join(history_parts)
    full_prompt += f"\n\n鐢ㄦ埛闂: {user_message}\n鍥炵瓟(涓枃,寮曠敤鏉ユ簮璇锋爣娉ㄧ紪鍙?:"

    client = _get_default_client()
    answer_chunks = []
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3,
                max_tokens=2000,
                stream=True,
            )
            async for chunk in response:
                token = chunk.choices[0].delta.content or ""
                if token:
                    answer_chunks.append(token)
                    if on_token_callback:
                        await on_token_callback(token)
            answer = "".join(answer_chunks).strip()
            if answer:
                return answer
        except Exception as e:
            print(f"[rag] LLM stream call failed (attempt {attempt + 1}): {e}")
            if attempt == 2:
                print("[rag] Streaming failed, falling back to non-stream chat...")
                answer = await _llm_chat(full_prompt, max_tokens=2000, temperature=0.3)
                if answer:
                    if on_token_callback:
                        await on_token_callback(answer)
                    return answer
            await asyncio.sleep(1.5)

    return KNOWLEDGE_INSUFFICIENT


# 鈹€鈹€ 涓绘祦绋?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


SYSTEM_PROMPT_TEMPLATE = """你是一个专业的游戏信息问答助手。你配备了工具，可以检索本地游戏知识库、搜索互联网或保存用户的长期记忆。

【你的核心目标】
准确回答用户的游戏相关问题。

【工具使用说明】
你可以使用以下工具：
1. `search_knowledge_base`: 优先使用此工具在本地数据库查找关于游戏的机制、故事、角色等详情。
2. `search_web`: 如果本地知识库没有提供足够的答案，或者用户询问最新的资讯，请使用此工具搜索互联网。
3. `save_user_preference`: 如果用户提到他们喜欢某类游戏，或者有什么特定的习惯/偏好，请使用此工具记录下来。

【回答要求】
- 当你给出最终回答时，如果你使用了任何工具检索到了信息，请使用 [来源标题](URL) 或 [来源N] 的格式标注引用。
- 请始终用中文回答。
- 如果多次尝试仍未找到信息，请诚实地告诉用户。

【关于当前用户的长期记忆】
以下是你之前了解到的关于当前用户的事实：
{user_memory_context}
请在回答时参考这些偏好。"""

async def rag_query(
    game_name: str,
    user_message: str,
    conversation_id: str,
    progress_callback=None,
) -> dict:
    """Agent loop handling tools, cache, and long-term memory."""
    async def _report(stage: str, message: str, content: str = None) -> None:
        if progress_callback is not None:
            try:
                await progress_callback(stage, message, content)
            except Exception:
                pass

    # 1. 妫€鏌ョ簿纭紦瀛?    query_hash = _get_query_hash(user_message, game_name)
    exact_hit = db.get_exact_query_cache(user_message, game_name)
    if exact_hit:
        await _report("cached", "鍛戒腑绮剧‘缂撳瓨锛岀洿鎺ヨ繑鍥炪€?)
        db.save_message(conversation_id, "user", user_message, None)
        db.save_message(conversation_id, "assistant", exact_hit, "[]")
        db.update_conversation_timestamp(conversation_id)
        return {"answer": exact_hit, "sources": [], "conversation_id": conversation_id}

    # 2. 妫€鏌ヨ涔夌紦瀛?    query_vec = encode_text(user_message)
    semantic_hit = db.get_semantic_query_cache(query_vec, game_name, threshold=0.85)
    if semantic_hit:
        await _report("cached", "鍛戒腑璇箟缂撳瓨锛岀洿鎺ヨ繑鍥炪€?)
        db.save_message(conversation_id, "user", user_message, None)
        db.save_message(conversation_id, "assistant", semantic_hit, "[]")
        db.update_conversation_timestamp(conversation_id)
        return {"answer": semantic_hit, "sources": [], "conversation_id": conversation_id}

    # 3. 鍑嗗 Agent 杩愯
    db.save_message(conversation_id, "user", user_message, None)
    history_msgs = db.get_messages(conversation_id, limit=settings.max_history_messages)
    
    # 鎻愬彇闀挎湡璁板繂
    memories = db.get_user_memories(query_vec, top_k=5)
    memory_text = "\n".join([f"- {m}" for m in memories]) if memories else "鏆傛棤璁板綍銆?
    
    system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{user_memory_context}", memory_text)
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history_msgs:
        if msg["content"] == user_message and msg["role"] == "user":
            continue # Already added as current message
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_message})

    max_loops = 5
    loop_count = 0
    final_answer = ""
    
    while loop_count < max_loops:
        loop_count += 1
        
        try:
            response = await _chat_with_fallback(
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.4,
                report_callback=_report
            )
        except Exception as e:
            final_answer = f"澶фā鍨嬭皟鐢ㄥけ璐ワ紝宸插皾璇曢檷绾ф満鍒跺叏閮ㄥけ璐? {str(e)}"
            break
            
        choice = response.choices[0]
        msg = choice.message
        
        messages.append(msg)
        
        if msg.tool_calls:
            # Execute tools
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                await _report("tool_call", f"姝ｅ湪浣跨敤宸ュ叿锛歿fn_name}...")
                print(f"[agent] Calling tool {fn_name} with args {args}")
                
                tool_result = await execute_tool(fn_name, args)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_result,
                })
        else:
            # Final answer
            final_answer = msg.content or ""
            break

    if loop_count >= max_loops and not final_answer:
        final_answer = "鎬濊€冭繃绋嬭繃闀匡紝鏈兘寰楀嚭鏈€缁堢粨璁恒€?

    # Stream the final answer if needed (we'll just report it as a chunk for now)
    await _report("generating", "鐢熸垚瀹屾瘯銆?, content=final_answer)

    # Cache the result
    db.set_query_cache(query_hash, game_name, final_answer, user_message, query_vec)

    # Save to db
    db.save_message(conversation_id, "assistant", final_answer, "[]")
    db.update_conversation_timestamp(conversation_id)

    # Auto title for new conversation
    conv = db.get_conversation(conversation_id)
    if conv and conv.get("title") == "New Conversation":
        new_title = user_message[:30] + ("..." if len(user_message) > 30 else "")
        db.update_conversation(conversation_id, new_title)

    return {"answer": final_answer, "sources": [], "conversation_id": conversation_id}

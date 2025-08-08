import os, re, json, time
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# ------------------ Config ------------------
load_dotenv()
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "40"))   # Prompt 1 分批大小
MAX_RETRY = 3
client = OpenAI()
# --------------------------------------------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def split_batches(items: List[Dict], n: int) -> List[List[Dict]]:
    return [items[i:i+n] for i in range(0, len(items), n)]

def extract_prompts(txt: str) -> Dict[str, str]:
    # 兼容我们给你的 llm4tg_prompts.txt 的格式
    def grab(title):
        m = re.search(rf"### {re.escape(title)}([\s\S]*?)(?=###|\Z)", txt)
        return (m.group(1).strip() if m else "")
    return {
        "p1": grab("Prompt 1 — Node Role Classification") or grab("Prompt 1"),
        "p2": grab("Prompt 2 — Anomaly Pattern Explanation") or grab("Prompt 2"),
        "p3": grab("Prompt 3 — Decentralization Snapshot Summary") or grab("Prompt 3"),
    }

def ask_llm(messages: List[Dict]) -> str:
    last_err = None
    for _ in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=messages,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err

def run_prompt1(nodes_jsonl_path: str, edges_csv_path: str, task_name: str, prompt_text: str, out_json_path: str):
    # Prompt 1 可能 token 比较大，对 nodes.jsonl 分批
    nodes = read_jsonl(nodes_jsonl_path)
    edges_csv = read_text(edges_csv_path)  # 只当参考，通常不会太大

    batches = split_batches(nodes, BATCH_SIZE)
    all_results = []
    for i, batch in enumerate(batches, 1):
        batch_blob = "\n".join(json.dumps(x, ensure_ascii=False) for x in batch)
        messages = [
            {"role":"system","content":"你是区块链交易网络分析助理，请严格按任务要求输出 JSON（id, role, confidence, rationale）。"},
            {"role":"user","content":f"[Task] {task_name} / Prompt 1 - Node Role Classification"},
            {"role":"user","content":f"以下为本批节点画像（JSONL）:\n```\n{batch_blob}\n```"},
            {"role":"user","content":f"以下为 Top100 诱导边（CSV，仅作为辅助，不必全用）:\n```\n{edges_csv}\n```"},
            {"role":"user","content":prompt_text}
        ]
        print(f"Running Prompt 1 batch {i}/{len(batches)} ...")
        content = ask_llm(messages)
        # 尝试解析返回的 JSON；如果模型返回了 markdown 代码块，做一下清洗
        content_clean = content.strip().strip("`").strip()
        # 保存分批原文，便于排错
        with open(out_json_path.replace(".json", f".part{i}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(content)
        try:
            result = json.loads(content_clean)
            if isinstance(result, list):
                all_results.extend(result)
            else:
                # 有的模型会包一层 {"items":[...]}
                all_results.extend(result.get("items", []))
        except Exception:
            # 解析失败就把原字符串塞进去，也不丢数据
            all_results.append({"_raw": content})

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[OK] Prompt 1 merged output -> {out_json_path}")

def run_prompt2(nodes_jsonl_path: str, edges_csv_path: str, task_name: str, prompt_text: str, out_txt_path: str):
    nodes_blob = read_text(nodes_jsonl_path)
    edges_blob = read_text(edges_csv_path)
    messages = [
        {"role":"system","content":"你是区块链交易网络分析助理，请输出简洁要点（每条≤60字）。"},
        {"role":"user","content":f"[Task] {task_name} / Prompt 2 - Anomaly Pattern Explanation"},
        {"role":"user","content":f"节点画像（JSONL，Top100）如下：\n```\n{nodes_blob}\n```"},
        {"role":"user","content":f"Top100 诱导边（CSV）如下：\n```\n{edges_blob}\n```"},
        {"role":"user","content":prompt_text}
    ]
    print("Running Prompt 2 ...")
    content = ask_llm(messages)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Prompt 2 -> {out_txt_path}")

def run_prompt3(nodes_jsonl_path: str, edges_csv_path: str, task_name: str, prompt_text: str, out_txt_path: str):
    # 直接给全量 Top100 节点画像/边 + Prompt 3
    nodes_blob = read_text(nodes_jsonl_path)
    edges_blob = read_text(edges_csv_path)
    messages = [
        {"role":"system","content":"你是区块链交易网络分析助理，请输出120-180字的执行摘要。"},
        {"role":"user","content":f"[Task] {task_name} / Prompt 3 - Decentralization Snapshot Summary"},
        {"role":"user","content":f"节点画像（JSONL，Top100）如下：\n```\n{nodes_blob}\n```"},
        {"role":"user","content":f"Top100 诱导边（CSV）如下：\n```\n{edges_blob}\n```"},
        {"role":"user","content":prompt_text}
    ]
    print("Running Prompt 3 ...")
    content = ask_llm(messages)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Prompt 3 -> {out_txt_path}")

def main():
    # 路径
    nodes_rwfb = "llm4tg_nodes_top100_rwfb.jsonl"
    edges_rwfb = "llm4tg_edges_top100_rwfb.csv"
    nodes_cet  = "llm4tg_nodes_top100_cetras_fast.jsonl"
    edges_cet  = "llm4tg_edges_top100_cetras_fast.csv"
    prompts_txt = read_text("llm4tg_prompts.txt")
    P = extract_prompts(prompts_txt)

    # ----- RWFB -----
    run_prompt1(nodes_rwfb, edges_rwfb, "RWFB", P["p1"], "rwfb_role_predictions.json")
    run_prompt2(nodes_rwfb, edges_rwfb, "RWFB", P["p2"], "rwfb_anomaly_explained.txt")
    run_prompt3(nodes_rwfb, edges_rwfb, "RWFB", P["p3"], "rwfb_decentralization_summary.txt")

    # ----- CETraS-like -----
    run_prompt1(nodes_cet,  edges_cet,  "CETraS-like", P["p1"], "cetras_role_predictions.json")
    run_prompt2(nodes_cet,  edges_cet,  "CETraS-like", P["p2"], "cetras_anomaly_explained.txt")
    run_prompt3(nodes_cet,  edges_cet,  "CETraS-like", P["p3"], "cetras_decentralization_summary.txt")

if __name__ == "__main__":
    main()

import iris
from sqlalchemy import create_engine,text
import config as cfg
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
emb_model = SentenceTransformer(os.getenv("EMB_MODEL"))

def get_iris_connection():
    return iris.connect(
        f"{cfg.IRIS_HOST}:{cfg.IRIS_PORT}/{cfg.IRIS_NAMESPACE}",
        cfg.IRIS_USER,
        cfg.IRIS_PASSWORD,
        sharedmemory=False
    )

def get_irispy():
    return iris.createIRIS(get_iris_connection())

def get_sqlalchemy_engine():
    url = f"iris://{cfg.IRIS_USER}:{cfg.IRIS_PASSWORD}@{cfg.IRIS_HOST}:{cfg.IRIS_PORT}/{cfg.IRIS_NAMESPACE}"
    return create_engine(url)

def upsert_graph_content(irispy, docid, title, abstract, url, published, authors, global_name="GraphContent"):
    irispy.set(title, global_name, docid, "title")
    irispy.set(abstract, global_name, docid, "abstract")
    irispy.set(url, global_name, docid, "url")
    irispy.set(published, global_name, docid, "published")
    irispy.set(authors, global_name, docid, "authors")

def upsert_graph_relations(irispy, docid, source, source_type, target, target_type, relation, global_name="GraphRelations"):
    irispy.set(source_type, global_name, docid, "Node", source)
    irispy.set(target_type, global_name, docid, "Node", target)
    irispy.set(relation, global_name, docid, "Edge", source, target)

def ensure_schema(engine, drop_first=False):
    from sqlalchemy import text
    with engine.connect() as conn:
        with conn.begin():
            if drop_first:
                conn.execute(text("DROP TABLE IF EXISTS paper_content"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS paper_content (
                    docid VARCHAR(255),
                    title VARCHAR(255),
                    abstract VARCHAR(2000),
                    url VARCHAR(255),
                    published VARCHAR(255),
                    authors VARCHAR(255),
                    combined VARCHAR(10000),
                    paper_vector VECTOR(FLOAT, 384)
                )
            """))

def create_hnsw_index(engine):
    from sqlalchemy import text
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text("""
                CREATE INDEX HNSWIndex
                ON TABLE paper_content (paper_vector)
                AS HNSW(Distance='DotProduct')
            """))


def search_papers(engine, search_vector, top_k):
    sql = text(f"""
        SELECT TOP {top_k} combined
        FROM paper_content
        ORDER BY VECTOR_DOT_PRODUCT(paper_vector, TO_VECTOR(:search_vector)) DESC
    """)
    with engine.connect() as conn:
        with conn.begin():
            rows = conn.execute(sql, {"search_vector": str(search_vector)}).fetchall()

    # Flatten 1-element tuples into a list of strings
    flattened = [r[0] for r in rows]
    return flattened



def send_to_llm(messages, **kwargs):
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=messages,
        temperature=float(os.getenv("OPENAI_TEMPERATURE")),
        **kwargs
    )
    return completion


def format_graph_context(combined_results: dict) -> str:
    """Format combined graph results into readable structured text for the LLM."""
    papers = combined_results.get("papers", [])
    graphs = combined_results.get("graphs", [])
    graph_by_id = {int(g["doc_id"]): g for g in graphs}

    sections = []
    for p in papers:
        doc_id = int(p.get("doc_id", 0))
        parts = []
        if p.get("title"):
            parts.append(f'Title: "{p["title"]}"')
        if p.get("authors"):
            parts.append(f"Authors: {p['authors']}")
        if p.get("published"):
            parts.append(f"Published: {p['published']}")
        if p.get("abstract"):
            parts.append(f"Abstract: {p['abstract'][:500]}")

        g = graph_by_id.get(doc_id)
        if g:
            topics = [n["name"] for n in g.get("nodes", []) if n.get("type") == "topic"]
            if topics:
                parts.append(f"Topics covered: {', '.join(topics)}")
            edges = g.get("edges", [])
            if edges:
                rel_strs = [
                    f"    {e['source']} --[{e.get('relation', '')}]--> {e['target']}"
                    for e in edges[:10]
                ]
                parts.append("Graph relations:\n" + "\n".join(rel_strs))

        sections.append("\n".join(parts))

    return "\n\n---\n\n".join(sections)


def fix_bullet_lines(lines):
    """Split lines where the LLM packed multiple ' - ' items onto one line."""
    result = []
    for line in lines:
        if line.startswith('- '):
            result.append(line)
        elif ' - ' in line:
            parts = [p.strip() for p in line.split(' - ') if p.strip()]
            for part in parts:
                result.append('- ' + part)
        else:
            result.append(line)
    return result


def llm_answer_rag(batch, query, cutoff=True):

    prompt_text = (
        "You are a research assistant. Answer the question using ONLY the text passages "
        "retrieved by semantic similarity search below.\n"
        "IMPORTANT LIMITATION: You only have access to flat text snippets. You have NO knowledge "
        "of graph structure, topic connections, paper relationships, citation links, or which "
        "topics appear in the most papers. If the question asks about graph structure or "
        "connections between papers/topics/authors, respond with a single bullet: "
        "'- RAG cannot answer this — it has no graph structure, only flat text snippets.'\n"
        "STRICT FORMAT RULES:\n"
        "  - Use bullet points only. Every bullet starts with '- ' on its OWN LINE.\n"
        "  - Never put two bullets on the same line.\n"
        "  - Never write a prose paragraph.\n"
        "  - If listing papers: format each as: - Title (year) — First Author et al.\n"
        "  - If explaining concepts: one bullet per key point.\n"
        "Do not invent author names, paper titles, or relationships not present in the text.\n\n"
        "Question: {question}\n\n"
        "Retrieved text passages:\n{context}\n\n"
        "Answer:"
    )

    prompt = prompt_text.format(question=query, context=batch)
 
    messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    completion = send_to_llm(messages)
    response = completion.choices[0].message.content
    answer_lines = [line.strip() for line in response.split('\n') if line.strip()]
    return fix_bullet_lines(answer_lines)


_GRAPH_STRUCTURAL_RE = re.compile(
    r'\b(connect|connects|most papers|most connected|topic.*graph|graph.*topic'
    r'|which topic|what topic|top topic|central topic'
    r'|link.*paper|paper.*link|graph structure|graph relation)\b',
    re.IGNORECASE,
)

def ask_question_rag(query: str, engine, emb_model, top_k: int = 5):

    if _GRAPH_STRUCTURAL_RE.search(query):
        return "- RAG cannot answer this — it only has flat text snippets with no knowledge of graph structure, topic connections, or which topics link the most papers."

    search_vector = emb_model.encode(query, normalize_embeddings=True).tolist()
    results = search_papers(engine, search_vector, top_k)
    response = llm_answer_rag(results, query, True)
    if isinstance(response, list):
        response = "\n".join(response)
    return response


# Graphrag
def get_graph_for_doc(doc_id: int, iris_handle, global_name="^GraphRelations"):
    nodes = []
    for name, node_type in iris_handle.iterator(global_name, doc_id, "Node"):
        nodes.append({"name": name, "type": node_type})


    edges = []
    for src, _ in iris_handle.iterator(global_name, doc_id, "Edge"):
        for dst, rel in iris_handle.iterator(global_name, doc_id, "Edge", src):
            edges.append({"source": src, "target": dst, "relation": rel})

    return {"doc_id": doc_id, "nodes": nodes, "edges": edges}

def search_papers_id(engine, search_vector,top_k):

    sql = text(f"""
         SELECT TOP {top_k} docid FROM paper_content ORDER BY VECTOR_DOT_PRODUCT(paper_vector, TO_VECTOR(:search_vector)) DESC
    """)
    with engine.connect() as conn:
        with conn.begin():
            resultsID = conn.execute(sql, {"search_vector": str(search_vector)}).fetchall()
    results = [row[0] for row in resultsID]
    return results

def get_graphs_for_docs(doc_ids, iris_handle, global_name="^GraphRelations"):
    graphs = []
    for doc_id in doc_ids:
        # ensure int (in case SQL returns strings)
        gid = int(doc_id)
        graphs.append(get_graph_for_doc(gid, iris_handle, global_name))
    return graphs

def get_content_for_docs(doc_ids, irispy, global_name="^GraphContent"):

    results = []
    fields = ["title", "abstract", "authors", "published", "url"]

    for doc_id in doc_ids:
        doc_data = {"doc_id": int(doc_id)}
        for field in fields:
            value = irispy.get(global_name, doc_id, field)
            if value is not None:
                doc_data[field] = str(value)
        results.append(doc_data)

    return results


def llm_answer_graphrag(batch, query, cutoff=True):

    # Format raw combined_results dict into clean readable text
    if isinstance(batch, dict):
        graph_context = format_graph_context(batch)
    else:
        graph_context = str(batch)

    prompt_text = (
        "You are a research assistant with access to a structured academic knowledge graph.\n"
        "Each paper is linked to its authors (AUTHORED relation) and the topics it covers (COVERS relation).\n"
        "STRICT FORMAT RULES:\n"
        "  - Use bullet points only. Every bullet starts with '- ' on its OWN LINE.\n"
        "  - Never put two bullets on the same line.\n"
        "  - Never write a prose paragraph.\n"
        "  - If listing papers: format each as: - Title (year) — First Author et al.\n"
        "  - If listing authors: format each as: - Author Name (N papers)\n"
        "  - If explaining concepts/improvements: one bullet per key point; "
        "cite a paper in parentheses only when directly relevant.\n"
        "\nQuestion: {question}\n\n"
        "Knowledge graph context (papers, authors, topics, and their relations):\n"
        "{graph_context}\n\n"
        "Answer:"
    )

    prompt = prompt_text.format(question=query, graph_context=graph_context)
 
    messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    completion = send_to_llm(messages)
    response = completion.choices[0].message.content
 
    answer_lines = [line.strip() for line in response.split('\n') if line.strip()]
    return fix_bullet_lines(answer_lines)

def prepare_combined_results(query, engine, emb_model, iris_handle, top_k=5):
    search_vector = emb_model.encode(query, normalize_embeddings=True).tolist()
    doc_ids = search_papers_id(engine, search_vector, top_k=top_k)
    return {
        "doc_ids": doc_ids,
        "papers": get_content_for_docs(doc_ids, iris_handle),
        "graphs": get_graphs_for_docs(doc_ids, iris_handle)
    }


def ask_question_graphrag(query: str, engine, emb_model, irispy, top_k: int = 5):
    combined_results = prepare_combined_results(query, engine, emb_model, irispy, top_k=top_k)
    response = llm_answer_graphrag(combined_results, query, True)
    if isinstance(response, list):
        response = " ".join(response)
    return response


def combine_graphs(graphs):
    nodes_map = {}
    edges_set = set()
    for g in graphs:
        for n in g["nodes"]:
            key = (n["name"], n.get("type", ""))
            nodes_map[key] = {"id": n["name"], "type": n.get("type", "")}
        for e in g["edges"]:
            edge_key = (e["source"], e["target"], e.get("relation",""))
            edges_set.add(edge_key)
    nodes = list(nodes_map.values())
    links = [{"source": s, "target": t, "relation": r} for (s, t, r) in edges_set]
    return {"nodes": nodes, "links": links}

#Agent
def get_papers_by_author(
    author: str,
    iris_handle,
    relations_global="^GraphRelations",
    content_global="^GraphContent",
    include_content=True,
    case_insensitive=True,
):
    def _eq(a, b):
        return a.lower() == b.lower() if case_insensitive else a == b

    results = []

    # iterate all docIds at the top level
    for doc_id, _ in iris_handle.iterator(relations_global):
        # iterate edge sources for this doc
        for src, _ in iris_handle.iterator(relations_global, doc_id, "Edge"):
            if not _eq(str(src), author):
                continue
            # iterate destinations; filter on relation == "AUTHORED"
            for dst, rel in iris_handle.iterator(relations_global, doc_id, "Edge", src):
                if str(rel).upper() != "AUTHORED":
                    continue

                item = {
                    "doc_id": int(doc_id),
                    "author": str(src),
                    "title": str(dst)
                }

                if include_content:
                    # pull extra fields from ^GraphContent
                    for f in ("title", "abstract", "authors", "published", "url"):
                        val = iris_handle.get(content_global, doc_id, f)
                        if val is not None:
                            item[f] = str(val)

                results.append(item)

    return results

def get_papers_by_topic(
    irispy,
    topic,
    relations_global="^GraphRelations",
    content_global="^GraphContent",
    edge_root="Edge",
    require_value="COVERS",
    case_insensitive=True,
    include_content=True   # just add this

):


    def _eq(a, b):
        return a.lower() == b.lower() if case_insensitive else a == b

    results = []

    # iterate all doc_ids
    for doc_id, _ in irispy.iterator(relations_global):
        for paper, _ in irispy.iterator(relations_global, doc_id, edge_root):
            for dst_topic, relation in irispy.iterator(relations_global, doc_id, edge_root, paper):
                if not _eq(str(dst_topic), topic):
                    continue
                if require_value and str(relation).upper() != str(require_value).upper():
                    continue

                # start with relation info
                item = {
                    "doc_id": int(doc_id),
                    "paper": str(paper),
                    "topic": str(dst_topic),
                    "relation": str(relation),
                }

                # add paper metadata if available
                for f in ("title", "abstract", "authors", "published", "url"):
                    v = irispy.get(content_global, doc_id, f)
                    if v is not None:
                        item[f] = str(v)

                results.append(item)

    return results


def get_top_authors_by_paper_count(
    irispy,
    limit: int = 10,
    relations_global: str = "^GraphRelations",
    content_global: str = "^GraphContent",
    edge_root: str = "Edge",
    authored_value: str = "AUTHORED",
    case_insensitive: bool = True,
    dedup: bool = True,
):

    counts = {}           
    seen_pairs = {}       

    for doc_id, _ in irispy.iterator(relations_global):
        for author, _ in irispy.iterator(relations_global, doc_id, edge_root):
            for paper, rel in irispy.iterator(relations_global, doc_id, edge_root, author):
                if str(rel).upper() != authored_value.upper():
                    continue

                key = (str(author).lower() if case_insensitive else str(author))

                if dedup:
                    sp = seen_pairs.setdefault(key, set())
                    pair = (int(doc_id), str(paper))
                    if pair in sp:
                        continue
                    sp.add(pair)

                entry = counts.setdefault(key, {"author": str(author), "count": 0, "papers": []})
                entry["count"] += 1

                # paper details from ^GraphContent
                paper_info = {
                    "doc_id": int(doc_id),
                    "paper_node": str(paper),  # name used in the Edge subscript
                }
                for f in ("title", "abstract", "authors", "published", "url"):
                    v = irispy.get(content_global, doc_id, f)
                    if v is not None:
                        paper_info[f] = str(v)

                entry["papers"].append(paper_info)

    # sort and trim
    items = sorted(counts.values(), key=lambda x: (-x["count"], x["author"].lower()))
    return items[:max(1, int(limit))]


def get_top_topics_by_paper_count(
    irispy,
    limit: int = 10,
    relations_global: str = "^GraphRelations",
    edge_root: str = "Edge",
    covers_value: str = "COVERS",
):
    """Count how many distinct papers cover each topic and return the top N."""
    topic_doc_ids = {}

    for doc_id, _ in irispy.iterator(relations_global):
        for src, _ in irispy.iterator(relations_global, doc_id, edge_root):
            for dst, rel in irispy.iterator(relations_global, doc_id, edge_root, src):
                if str(rel).upper() != covers_value.upper():
                    continue
                topic = str(dst)
                topic_doc_ids.setdefault(topic, set()).add(int(doc_id))

    ranked = sorted(topic_doc_ids.items(), key=lambda x: len(x[1]), reverse=True)
    return [{"topic": t, "paper_count": len(ids)} for t, ids in ranked[:max(1, int(limit))]]


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_top_authors_by_paper_count",
            "description": "Return the top authors ranked by number of authored papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 100}
                },
                "required": []
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_papers_by_author",
            "description": "Return papers authored by the specified author, optionally with metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "author": {"type": "string", "description": "Author full name"},
                    "include_content": {"type": "boolean", "default": True}
                },
                "required": ["author"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_papers_by_topic",
            "description": "Return papers related to the specified topic, with relations and metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "include_content": {"type": "boolean", "default": True}
                },
                "required": ["topic"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_topics_by_paper_count",
            "description": "Return the top topics ranked by how many papers cover them. Use this for questions about which topics appear most, connect the most papers, or are most common in the graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
                },
                "required": []
            }
        },
    },
]

# -------- routing instruction --------
ROUTER_SYSTEM = (
    "You are a routing assistant for a graph-of-papers. "
    "Choose and call the correct tool(s). Examples:\n"
    "- 'who has written the most papers' -> call get_top_authors_by_paper_count(limit=5)\n"
    "- 'what did Harry Shomer write' -> call get_papers_by_author(author='Harry Shomer')\n"
    "- 'papers about Knowledge Graphs' -> call get_papers_by_topic(topic='Knowledge Graphs')\n"
    "- 'what topics connect the most papers' -> call get_top_topics_by_paper_count(limit=10)\n"
    "- 'which topics appear most in the graph' -> call get_top_topics_by_paper_count(limit=10)\n"
    "After tool results come back, format your answer as a bulleted list (- item per topic or author)."
)

##safe version
# def run_agent(user_query: str, irispy, limit_default: int = 5, debug: bool = True):
#     messages = [
#         # {"role": "system", "content": "Route to the correct tool. Answer concisely after tools."},
#         {"role": "system", "content": ROUTER_SYSTEM},

#         {"role": "user", "content": user_query},
#     ]

#     def call_tools(name: str, args: dict):
#         if debug:
#             print(f"[Agent] → {name}({args})")
#         if name == "get_top_authors_by_paper_count":
#             return get_top_authors_by_paper_count(irispy, limit=int(args.get("limit", limit_default)))
#         if name == "get_papers_by_author":
#             return get_papers_by_author(args["author"], irispy, include_content=bool(args.get("include_content", True)))
#         if name == "get_papers_by_topic":
#             return get_papers_by_topic(irispy, args["topic"],  include_content=bool(args.get("include_content", True)))
#         return {"error": f"unknown tool {name}"}

#     for step in range(3):
#         if debug: print(f"[Agent] step {step+1}")

#         # Ask model what to do
#         resp = send_to_llm(messages, tools=tools)

#         msg = resp.choices[0].message
#         tool_calls = msg.tool_calls or []

#         # IMPORTANT: append the assistant message with tool_calls BEFORE tool outputs
#         if tool_calls:
#             # convert tool_calls to plain dicts (SDK objects aren’t JSON serializable)
#             tc_dicts = [{
#                 "id": tc.id,
#                 "type": "function",
#                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}
#             } for tc in tool_calls]

#             messages.append({
#                 "role": "assistant",
#                 "content": msg.content or "",
#                 "tool_calls": tc_dicts
#             })

#             # Execute each tool and append the tool result
#             for tc in tool_calls:
#                 fn_name = tc.function.name
#                 fn_args = json.loads(tc.function.arguments or "{}")
#                 result = call_tools(fn_name, fn_args)

#                 if debug:
#                     preview = json.dumps(result, ensure_ascii=False)[:200]
#                     print(f"[Agent] result: {preview}...\n")

#                 messages.append({
#                     "role": "tool",
#                     "tool_call_id": tc.id,
#                     "name": fn_name,
#                     "content": json.dumps(result, ensure_ascii=False)
#                 })

#             # loop continues; model will now see the tool outputs and (usually) produce final answer
#             continue

#         # No tool calls → final answer
#         if debug: print("[Agent] ✓ Final answer")
#         return msg.content

#     return "Agent stopped: max steps reached."

# def run_agent(user_query, irispy, limit_default=5, debug=True, return_context=False):
#     messages = [{"role":"system","content":ROUTER_SYSTEM},{"role":"user","content":user_query}]
#     last_tool, last_payload = None, None

#     def call_tools(name, args):
#         nonlocal last_tool, last_payload
#         if name == "get_top_authors_by_paper_count":
#             result = get_top_authors_by_paper_count(irispy, limit=int(args.get("limit", limit_default)))
#         elif name == "get_papers_by_author":
#             result = get_papers_by_author(args["author"], irispy, include_content=bool(args.get("include_content", True)))
#         elif name == "get_papers_by_topic":
#             result = get_papers_by_topic(irispy, args["topic"], include_content=bool(args.get("include_content", True)))
#         else:
#             result = {"error": f"unknown tool {name}"}
#         last_tool, last_payload = name, result
#         return result

#     for _ in range(3):
#         resp = send_to_llm(messages, tools=tools)
#         msg = resp.choices[0].message
#         tcs = msg.tool_calls or []
#         if tcs:
#             tc_dicts = [{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in tcs]
#             messages.append({"role":"assistant","content":msg.content or "", "tool_calls": tc_dicts})
#             for tc in tcs:
#                 result = call_tools(tc.function.name, json.loads(tc.function.arguments or "{}"))
#                 messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.function.name,"content":json.dumps(result, ensure_ascii=False)})
#             continue
#         # Final
#         return {"answer": msg.content, "tool": last_tool, "payload": last_payload} if return_context else msg.content

#     return {"answer":"Agent stopped: max steps reached.","tool":last_tool,"payload":last_payload} if return_context else "Agent stopped: max steps reached."

def run_agent(user_query, irispy, limit_default=5, debug=True, return_context=False):
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": user_query},
    ]

    last_tool = None
    last_payload = None

    def call_tools(name: str, args: dict):
        nonlocal last_tool, last_payload
        if debug:
            print(f"[Agent] → {name}({args})")
        if name == "get_top_authors_by_paper_count":
            result = get_top_authors_by_paper_count(irispy, limit=int(args.get("limit", limit_default)))
        elif name == "get_papers_by_author":
            result = get_papers_by_author(args["author"], irispy, include_content=bool(args.get("include_content", True)))
        elif name == "get_papers_by_topic":
            result = get_papers_by_topic(irispy, args["topic"], include_content=bool(args.get("include_content", True)))
        elif name == "get_top_topics_by_paper_count":
            result = get_top_topics_by_paper_count(irispy, limit=int(args.get("limit", 10)))
        else:
            result = {"error": f"unknown tool {name}"}
        last_tool, last_payload = name, result
        return result

    for step in range(3):
        if debug: print(f"[Agent] step {step+1}")
        resp = send_to_llm(messages, tools=tools)
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if tool_calls:
            tc_dicts = [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            } for tc in tool_calls]
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": tc_dicts})

            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")
                result = call_tools(fn_name, fn_args)
                if debug:
                    preview = json.dumps(result, ensure_ascii=False)[:200]
                    print(f"[Agent] result: {preview}...\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            continue

        # final answer
        if debug: print("[Agent] ✓ Final answer")
        if return_context:
            doc_ids = extract_doc_ids_from_tool(last_tool, last_payload)
            return {"answer": msg.content, "doc_ids": doc_ids or []}
        return msg.content

    if return_context:
        return {"answer": "Agent stopped: max steps reached.", "doc_ids": []}
    return "Agent stopped: max steps reached."


_AGGREGATION_RE = re.compile(
    r'\b(top|most|least|how many|count|number of|rank|who has the most|'
    r'who have the most|list all|which .* most|highest|lowest|best|worst)\b',
    re.IGNORECASE
)

_COMPARISON_RE = re.compile(
    r'\b(improve|improvement|better|advantage|benefit|differ|difference|'
    r'compare|comparison|over rag|vs rag|versus rag|over traditional|'
    r'graphrag.*rag|rag.*graphrag)\b',
    re.IGNORECASE
)

def classify_query_llm(user_query: str) -> str:
    """Keyword-based classifier — no LLM call needed."""
    if _COMPARISON_RE.search(user_query):
        return "comparison"
    return "aggregation" if _AGGREGATION_RE.search(user_query) else "general"


def graphrag_vs_rag_answer(irispy) -> str:
    """
    Pull real graph statistics and synthesise a concrete comparison answer
    that highlights what GraphRAG can do that RAG cannot.
    """
    topics = get_top_topics_by_paper_count(irispy, limit=5)
    authors = get_top_authors_by_paper_count(irispy, limit=3)

    # Count total nodes and edges across the whole graph
    total_papers = 0
    total_edges = 0
    for doc_id, _ in irispy.iterator("^GraphRelations"):
        total_papers += 1
        for src, _ in irispy.iterator("^GraphRelations", doc_id, "Edge"):
            for dst, _ in irispy.iterator("^GraphRelations", doc_id, "Edge", src):
                total_edges += 1

    top_topics_str = ", ".join(
        f"{t['topic']} ({t['paper_count']} papers)" for t in topics
    )
    top_authors_str = ", ".join(
        f"{a['author']} ({a['count']} papers)" for a in authors
    )

    prompt = (
        "You are a research assistant. Explain how GraphRAG improves over traditional RAG.\n"
        "Use the REAL graph statistics below to make the answer concrete and specific.\n"
        "STRICT FORMAT RULES:\n"
        "  - Use bullet points only. Every bullet starts with '- ' on its OWN LINE.\n"
        "  - Never put two bullets on the same line. Never write a prose paragraph.\n"
        "  - Each bullet should be a distinct advantage of GraphRAG over RAG.\n"
        "  - Weave in the real numbers and topic/author names to show graph power.\n\n"
        f"Real graph statistics from the knowledge graph:\n"
        f"  - {total_papers} papers indexed as graph nodes with {total_edges} relationships\n"
        f"  - Top topics by paper coverage: {top_topics_str}\n"
        f"  - Most prolific authors: {top_authors_str}\n\n"
        "Question: How does GraphRAG improve over traditional RAG systems?\n\n"
        "Answer (bullet points, each on its own line):"
    )

    messages = [{"role": "user", "content": prompt}]
    completion = send_to_llm(messages)
    response = completion.choices[0].message.content
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    return "\n".join(fix_bullet_lines(lines))
##safe version
# def ask_question_graphrag_agent(
#     user_query: str,
#     irispy,
#     combined_results=None,
#     engine=None,
#     emb_model=None,
#     top_k: int = 5,
#     debug: bool = True
# ):
#     qtype = classify_query_llm(user_query)
#     if debug:
#         print(f"[Router] classified as: {qtype}")

#     if qtype == "aggregation":
#         # route to tools
#         return run_agent(user_query, irispy, debug=debug)

#     # general Q&A over graph+papers:
#     if combined_results is None:
#         # compute if not provided (requires engine & emb_model)
#         if engine is None or emb_model is None:
#             raise ValueError("combined_results not provided; need engine and emb_model to compute it.")
#         combined_results = prepare_combined_results(user_query, engine, emb_model, irispy, top_k=top_k)

#     # answer using the already-prepared combined results
#     resp = llm_answer_graphrag(combined_results, user_query, True)
#     if isinstance(resp, list):
#         resp = " ".join(resp)
#     return resp

# def ask_question_graphrag_agent(
#     user_query: str,
#     irispy,
#     combined_results=None,
#     engine=None,
#     emb_model=None,
#     top_k: int = 5,
#     debug: bool = True,
#     return_context: bool = False
# ):
#     qtype = classify_query_llm(user_query)

#     if qtype == "aggregation":
#         ctx = run_agent(user_query, irispy, debug=debug, return_context=True)
#         doc_ids = extract_doc_ids_from_tool(ctx.get("tool"), ctx.get("payload"))
#         return {"answer": ctx["answer"], "doc_ids": doc_ids} if return_context else ctx["answer"]

#     # general Q&A over graph+papers
#     if combined_results is None:
#         if engine is None or emb_model is None:
#             raise ValueError("combined_results not provided; need engine and emb_model to compute it.")
#         combined_results = prepare_combined_results(user_query, engine, emb_model, irispy, top_k=top_k)

#     ans = llm_answer_graphrag(combined_results, user_query, True)
#     if isinstance(ans, list):
#         ans = " ".join(ans)
#     return {"answer": ans, "doc_ids": combined_results.get("doc_ids", [])} if return_context else ans


def ask_question_graphrag_agent(
    user_query: str,
    irispy,
    combined_results=None,
    engine=None,
    emb_model=None,
    top_k: int = 5,
    debug: bool = True,
    return_context: bool = False
):
    qtype = classify_query_llm(user_query)
    if debug:
        print(f"[Router] classified as: {qtype}")

    if qtype == "comparison":
        ans = graphrag_vs_rag_answer(irispy)
        if return_context:
            return {"answer": ans, "doc_ids": [], "method": "comparison"}
        return ans

    if qtype == "aggregation":
        ctx = run_agent(user_query, irispy, debug=debug, return_context=True)
        if return_context:
            ctx["method"] = "agent"
            return ctx
        return ctx["answer"]

    if combined_results is None:
        if engine is None or emb_model is None:
            raise ValueError("combined_results not provided; need engine and emb_model to compute it.")
        combined_results = prepare_combined_results(user_query, engine, emb_model, irispy, top_k=top_k)

    ans = llm_answer_graphrag(combined_results, user_query, True)
    if isinstance(ans, list):
        ans = "\n".join(ans)

    if return_context:
        return {"answer": ans, "doc_ids": combined_results.get("doc_ids", []), "method": "graph_general"}
    return ans


def extract_doc_ids_from_tool(tool_name, payload):
    ids = []
    if not payload:
        return ids
    try:
        if tool_name == "get_top_authors_by_paper_count":
            for entry in payload:
                for p in entry.get("papers", []):
                    if "doc_id" in p:
                        ids.append(int(p["doc_id"]))
        elif tool_name in ("get_papers_by_author", "get_papers_by_topic"):
            for item in payload:
                if "doc_id" in item:
                    ids.append(int(item["doc_id"]))
    except Exception:
        pass
    out, seen = [], set()
    for i in ids:
        if i not in seen:
            out.append(i); seen.add(i)
    return out



if __name__ == "__main__":
    try:
        irispy = get_irispy()
        print("Connected with IRIS API!")

        engine = get_sqlalchemy_engine()
        print("SQLAlchemy engine ready!")

    except Exception as e:
        print(f"Connection failed: {e}")
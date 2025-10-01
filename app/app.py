# app.py
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

from iris_db import (
    # IRIS & RAG helpers
    get_sqlalchemy_engine,
    get_irispy,
    search_papers_id,
    get_graphs_for_docs,
    combine_graphs,
    ask_question_rag,
    ask_question_graphrag_agent,
    emb_model as default_emb_model,
    send_to_llm,
    get_content_for_docs,          

)

load_dotenv()
# app = Flask(__name__, template_folder="templates")
app = Flask(__name__)


_engine = None
_emb_model = None
_irispy = None

EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def ensure_ready():
    """Lazy init heavyweight dependencies."""
    global _engine, _emb_model, _irispy

    if _engine is None:
        _engine = get_sqlalchemy_engine()

    if _emb_model is None:
        try:
            _emb_model = SentenceTransformer(EMB_MODEL_NAME)
        except Exception:
            # Fall back to the model instance created in iris_db.py
            _emb_model = default_emb_model

    if _irispy is None:
        try:
            _irispy = get_irispy()
        except Exception as e:
            _irispy = None
            print("IRIS native init failed:", repr(e))



@app.get("/")
@app.get("/rag")
def rag_page():
    # Render templates/rag.html
    return render_template("rag.html")


@app.post("/api/ask")
def api_ask():
    ensure_ready()  # engine + emb_model are required; IRIS native not required here
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top_k") or 5)

    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    try:
        answer = ask_question_rag(query, _engine, _emb_model, top_k=top_k)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.get("/graphrag")
def graphrag_page():
   
    return render_template("graphrag.html")

##safe version
# @app.post("/api/graphrag")
# def api_graphrag():
#     ensure_ready()
#     if _irispy is None:
#         return jsonify({"error": "IRIS native not initialized. Check IRIS_* env and get_irispy()."}), 500

#     data = request.get_json(force=True, silent=True) or {}
#     query = (data.get("query") or "").strip()
#     top_k = int(data.get("top_k") or 5)

#     if not query:
#         return jsonify({"error": "Missing 'query'"}), 400

#     try:
#         # Vector search to get doc IDs
#         sv = _emb_model.encode(query, normalize_embeddings=True).tolist()
#         doc_ids = search_papers_id(_engine, sv, top_k=top_k)

#         # Graph assembly
#         graphs = get_graphs_for_docs(doc_ids, _irispy)
#         merged_graph = combine_graphs(graphs)
#         content = get_content_for_docs(doc_ids, _irispy)
#         refs = [{"doc_id": c.get("doc_id"), "title": c.get("title"), "url": c.get("url")} for c in content]
#         # Pass minimal combined results to the agent
#         combined_results = {"doc_ids": doc_ids, "graphs": graphs}
#         answer = ask_question_graphrag_agent(
#             user_query=query,
#             irispy=_irispy,
#             combined_results=combined_results,
#             debug=False,
#         )

#         return jsonify({
#             "answer": answer,
#             "doc_ids": doc_ids,
#             "papers": refs,    
#             "graph": merged_graph
#         })
   
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.post("/api/graphrag")
# def api_graphrag():
#     ensure_ready()
#     if _irispy is None:
#         return jsonify({"error": "IRIS native not initialized. Check IRIS_* env and get_irispy()."}), 500

#     data = request.get_json(force=True, silent=True) or {}
#     query = (data.get("query") or "").strip()
#     top_k = int(data.get("top_k") or 5)

#     if not query:
#         return jsonify({"error": "Missing 'query'"}), 400

#     try:
#         # 1) vector search candidates (default)
#         sv = _emb_model.encode(query, normalize_embeddings=True).tolist()
#         vec_doc_ids = search_papers_id(_engine, sv, top_k=top_k)

#         # 2) ask agent and let it tell us doc_ids for aggregation
#         combined_results = {"doc_ids": vec_doc_ids}  # minimal pass-through
#         agent_out = ask_question_graphrag_agent(
#             user_query=query,
#             irispy=_irispy,
#             combined_results=combined_results,
#             engine=_engine,           # in case general path needs it
#             emb_model=_emb_model,     # "
#             top_k=top_k,
#             debug=False,
#             return_context=True
#         )

#         answer = agent_out.get("answer", "")
#         # Prefer agent's doc_ids if provided (aggregation); else fallback to vector doc_ids
#         doc_ids_for_graph = agent_out.get("doc_ids") or vec_doc_ids
#         # Optional: cap for UI sanity
#         doc_ids_for_graph = list(doc_ids_for_graph)[:30]

#         # 3) build graph & refs from the chosen doc ids
#         graphs = get_graphs_for_docs(doc_ids_for_graph, _irispy)
#         merged_graph = combine_graphs(graphs)
#         content = get_content_for_docs(doc_ids_for_graph, _irispy)
#         refs = [{"doc_id": c.get("doc_id"), "title": c.get("title"), "url": c.get("url")} for c in content]

#         return jsonify({
#             "answer": answer,
#             "doc_ids": doc_ids_for_graph,
#             "papers": refs,
#             "graph": merged_graph
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.post("/api/graphrag")
def api_graphrag():
    ensure_ready()
    if _irispy is None:
        return jsonify({"error": "IRIS native not initialized. Check IRIS_* env and get_irispy()."}), 500

    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top_k") or 5)
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    try:
        # vector search candidates (fallback if agent doesn't return ids)
        sv = _emb_model.encode(query, normalize_embeddings=True).tolist()
        vec_doc_ids = search_papers_id(_engine, sv, top_k=top_k)

        # let agent answer; for general it will compute full context; for aggregation it returns tool-driven doc_ids
        agent_out = ask_question_graphrag_agent(
            user_query=query,
            irispy=_irispy,
            combined_results=None,   # <-- let it compute full (papers+graphs) for general questions
            engine=_engine,
            emb_model=_emb_model,
            top_k=top_k,
            debug=False,
            return_context=True
        )

        answer = agent_out.get("answer", "")
        doc_ids_for_graph = agent_out.get("doc_ids") or vec_doc_ids
        doc_ids_for_graph = list(doc_ids_for_graph)[:30]

        graphs = get_graphs_for_docs(doc_ids_for_graph, _irispy)
        merged_graph = combine_graphs(graphs)
        content = get_content_for_docs(doc_ids_for_graph, _irispy)
        refs = [{"doc_id": c.get("doc_id"), "title": c.get("title"), "url": c.get("url")} for c in content]

        return jsonify({
            "answer": answer,
            "doc_ids": doc_ids_for_graph,
            "papers": refs,
            "graph": merged_graph
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.get("/llm")
def llm_page():
    
    return render_template("llm.html")


@app.post("/api/llm")
def api_llm():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing 'query'"}), 400

        resp = send_to_llm([{"role": "user", "content": query}])
        msg = resp.choices[0].message
        return jsonify({"answer": msg.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/llm-vs-rag")
def compare_page():
    return render_template("llm_vs_rag.html")


@app.get("/rag-vs-graphrag")
def rag_vs_graphrag_page():
    return render_template("rag_vs_graphrag.html")


if __name__ == "__main__":
    try:
        ensure_ready()
        print("Engine ready:", _engine is not None)
        print("Emb model:", EMB_MODEL_NAME)
        print("IRIS native ready:", _irispy is not None)
    except Exception as e:
        print("Startup warning:", e)

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)

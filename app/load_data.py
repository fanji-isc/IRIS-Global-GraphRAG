import csv
import iris
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import os


args = {
        'hostname': 'iris',
        'port': 1972,
        'namespace': 'USER',
        'username': '_SYSTEM',
        'password': 'SYS',
        'logfile': 'log.txt'
    }
conn = iris.connect(**args)
irispy = iris.createIRIS(conn)

print(os.getcwd())
print(os.path.exists("data/papers.csv"))
def load_papers(csv_path="data/papers.csv"):


    conn = iris.connect(**args)
    irispy = iris.createIRIS(conn)

    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            docid = row['docid']
            title = row['title']
            abstract = row['abstract']
            url = row['url']
            published = row['published']
            authors = row['authors']

            irispy.set(title,     "GraphContent", docid, "title")
            irispy.set(abstract,  "GraphContent", docid, "abstract")
            irispy.set(url,       "GraphContent", docid, "url")
            irispy.set(published, "GraphContent", docid, "published")
            irispy.set(authors,   "GraphContent", docid, "authors")

    conn.close()


def load_relations(csv_path="data/relations.csv"):
 
   
    conn = iris.connect(**args)
    irispy = iris.createIRIS(conn)

    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            docid = row['docid']
            source = row['source']
            source_type = row['sourcetype']
            target = row['target']
            target_type = row['targettype']
            relation = row['type']

            irispy.set(source_type, "GraphRelations", docid, "Node", source)
            irispy.set(target_type, "GraphRelations", docid, "Node", target)
            irispy.set(relation,    "GraphRelations", docid, "Edge", source, target)

    conn.close()


def embed_and_load_papers(
    csv_path="data/papers_combined.csv",
    conn_url="iris://_SYSTEM:SYS@iris:1972/USER",
    model_name="all-MiniLM-L6-v2"
):
    df = pd.read_csv(csv_path)

    if "combined" not in df.columns:
        df["combined"] = df.apply(
            lambda r: f"docid: {r['docid']} | title: {r['title']} | abstract: {r['abstract']} | "
                      f"url: {r['url']} | published: {r['published']} | authors: {r['authors']}",
            axis=1
        )

    emb_model = SentenceTransformer(model_name)
    embeddings = emb_model.encode(df['combined'].tolist(), normalize_embeddings=True)
    df["paper_vector"] = embeddings.tolist()

    engine = create_engine(conn_url)

    create_sql = """
        CREATE TABLE IF NOT EXISTS SQLUser.paper_content (
            docid VARCHAR(255),
            title VARCHAR(255),
            abstract VARCHAR(2000),
            url VARCHAR(255),
            published VARCHAR(255),
            authors VARCHAR(255),
            combined VARCHAR(10000),
            paper_vector VECTOR(FLOAT, 384)
        )
    """

    with engine.connect() as conn:
        with conn.begin(): 
            conn.execute(text(create_sql))

            for _, row in df.iterrows():
                insert_sql = text("""
                    INSERT INTO SQLUser.paper_content 
                    (docid, title, abstract, url, published, authors, combined, paper_vector) 
                    VALUES (:docid, :title, :abstract, :url, :published, :authors, :combined, TO_VECTOR(:paper_vector))
                """)
                conn.execute(insert_sql, {
                    "docid": row["docid"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "url": row["url"],
                    "published": row["published"],
                    "authors": row["authors"],
                    "combined": row["combined"],
                    "paper_vector": str(row["paper_vector"])
                })

    print("✅ Papers embedded and loaded into SQLUser.PAPER_CONTENT")



def create_vector_index(
    conn_url="iris://_SYSTEM:SYS@iris:1972/USER",
    table_name="paper_content",
    column_name="paper_vector",
    index_name="HNSWIndex"
):


    engine = create_engine(conn_url)

    sql = f"""
        CREATE INDEX {index_name}
        ON TABLE {table_name} ({column_name})
        AS HNSW(Distance='DotProduct')
    """

    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text(sql))
            print(f"✅ Created HNSW index '{index_name}' on {table_name}({column_name})")

if __name__ == "__main__":
    load_papers()
    load_relations()
    embed_and_load_papers()
    create_vector_index()

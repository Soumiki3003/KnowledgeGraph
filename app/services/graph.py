from contextlib import AbstractContextManager
import json
import logging
from pathlib import Path
import time
from typing import Callable
from neo4j import Session, ManagedTransaction
from pydantic_ai import Agent, Embedder

logger = logging.getLogger(__name__)


class GraphService:
    def __init__(
        self,
        session_factory: Callable[..., AbstractContextManager[Session]],
        embedder: Embedder,
        llm: Agent,
    ):
        self.__session_factory = session_factory
        self.__embedder = embedder
        self.__llm = llm

    def __read_json_graph(self, filepath: Path) -> dict:
        if not filepath.exists():
            raise FileNotFoundError(f"Graph JSON file not found: {filepath}")
        if not filepath.is_file():
            raise ValueError(f"Provided path is not a file: {filepath}")

        return json.loads(filepath.read_text("utf-8"))

    # === Flatten helper (to handle nested maps like progress_metric) ===
    def __flatten_props(self, props: dict) -> dict:
        flat = {}
        for k, v in props.items():
            if isinstance(v, dict):
                # flatten maps like progress_metric
                for subk, subv in v.items():
                    flat[f"{k}_{subk}"] = subv
            elif isinstance(v, list):
                # flatten lists of simple values
                if all(isinstance(i, (str, int, float, bool)) for i in v):
                    flat[k] = v
                elif all(isinstance(i, dict) for i in v):
                    # for list of dicts that are not question_prompts
                    flat[k] = [json.dumps(i, ensure_ascii=False) for i in v]
                else:
                    flat[k] = str(v)
            else:
                flat[k] = v
        return flat

    def __create_nodes_and_relationships(
        self,
        tx: ManagedTransaction,
        node: dict,
        parent_id: str | None = None,
    ):
        node_id = node.get("id")
        if not node_id:
            raise ValueError(f"Node is missing 'id' field: {node}")
        if not isinstance(node_id, str):
            raise TypeError(f"Node 'id' must be a string in node: {node}")

        logger.debug(f"Creating node and relationships for node ID: {node_id}")

        node_label = "Concept"
        if node_id.startswith("P"):
            node_label = "Procedure"
        elif node_id.startswith("A"):
            node_label = "Assessment"

        # ✅ Flatten everything except children & connections
        props = self.__flatten_props(
            {
                k: v
                for k, v in node.items()
                if k not in ["children", "connections", "question_prompts"]
            }
        )

        # Create node
        tx.run(
            f"""
            MERGE (n:{node_label} {{id:$id}})
            SET n += $props
            """,
            id=node_id,
            props=props,
        )

        # === If Assessment has question_prompts, make Question nodes ===
        if node_label == "Assessment" and "question_prompts" in node:
            for idx, q in enumerate(node["question_prompts"], start=1):
                if isinstance(q, dict):
                    q_text = q.get("question", "")
                else:
                    q_text = str(q)
                q_id = f"{node_id}-Q{idx}"
                tx.run(
                    """
                    MERGE (q:Question {id:$qid})
                    SET q.text = $text
                    WITH q
                    MATCH (a {id:$aid})
                    MERGE (a)-[:HAS_QUESTION]->(q)
                    """,
                    qid=q_id,
                    text=q_text,
                    aid=node_id,
                )

        # === Create HAS_CHILD relationship if nested ===
        if parent_id:
            tx.run(
                """
                MATCH (p {id:$parent_id}), (c {id:$child_id})
                MERGE (p)-[:HAS_CHILD]->(c)
                """,
                parent_id=parent_id,
                child_id=node_id,
            )

        # === Create connections between concepts ===
        for conn in node.get("connections", []):
            if not isinstance(conn, dict):
                raise TypeError(
                    f"Was expecting connection to be a dict in node {node_id}, got: {type(conn)}"
                )
            if "to" not in conn:
                raise ValueError(
                    f"Connection missing 'to' field in node {node_id}: {conn}"
                )

            conn_relation = str(conn["relation"])
            conn_to = str(conn["to"])

            logger.debug(
                f"Creating connection from {node_id} to {conn_to} with relation {conn_relation}"
            )
            tx.run(
                f"""
                MATCH (a {{id:$from_id}}), (b {{id:$to_id}})
                MERGE (a)-[:{conn_relation}]->(b)
                """,
                from_id=node_id,
                to_id=conn_to,
            )

        # === Recurse for children ===
        logger.debug(f"Recursing into child node of {node_id}")
        for child in node.get("children", []):
            if not isinstance(child, dict):
                raise TypeError(
                    f"Was expecting child to be a dict in node {node_id}, got: {type(child)}"
                )
            self.__create_nodes_and_relationships(tx, child, parent_id=node_id)

    def add_from_file(self, filepath: Path) -> None:
        logger.info(f"Loading graph from file: {filepath}")
        try:
            graph = self.__read_json_graph(filepath)
            with self.__session_factory() as session:
                session.execute_write(self.__create_nodes_and_relationships, graph)
            logger.info("Graph successfully loaded into Neo4j")
        except Exception as e:
            logger.error(f"Error loading graph into Neo4j: {e}")
            raise

    def reembed_nodes(self) -> None:
        logger.info("Re-embedding all nodes in Neo4j")
        with self.__session_factory() as session:
            query = """
            MATCH (n)
            WHERE n:Concept OR n:Procedure OR n:Assessment
            RETURN n.id AS id, n.name AS name, n.definition AS definition
            """
            records = list(session.run(query))
            logger.info(f"🔍 Found {len(records)} nodes to re-embed")
            for i, record in enumerate(records, 1):
                node_id = record["id"]
                text = f"{record['name']} {record.get('definition','')}"

                # ✅ no need for Gemini quota handling — this is local now
                vector = self.__embedder.embed_documents_sync(text).embeddings[0]

                session.run(
                    "MATCH (n {id:$id}) SET n.embedding=$embedding",
                    id=node_id,
                    embedding=vector,
                )

                logger.info(f"[{i}/{len(records)}] {node_id} re-embedded")
                time.sleep(0.5)  # gentle throttle to avoid Neo4j overload

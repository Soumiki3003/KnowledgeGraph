import logging
from typing import Any, Literal

from neo4j import ManagedTransaction, Session, unit_of_work
from neo4j_graphrag.generation import GraphRAG
from pydantic_ai import Embedder

from app import models, schemas
from app.utils import hash_string

from .auth import AuthService


class StudentService:
    __student_node_name = "Student"

    __trajectory_node_name = "StudentTrajectory"
    __trajectory_rel_name = "HAS_TRAJECTORY"
    __trajectory_prev_rel_name = "PREVIOUS_TRAJECTORY"

    def __init__(
        self,
        session: Session,
        embedder: Embedder,
        rag: GraphRAG,
        auth_service: AuthService,
        trajectory_vector_index_field: str,
        trajectory_full_text_index_field: str,
    ):
        self.__session = session
        self.__embedder = embedder
        self.__auth_service = auth_service
        self.__rag = rag
        self.__trajectory_vector_index_field = trajectory_vector_index_field
        self.__trajectory_full_text_index_field = trajectory_full_text_index_field
        self.__logger = logging.getLogger(__name__)

    def create_student(self, item: schemas.CreateStudent) -> models.Student:
        self.__logger.info(f"Creating new student with email: {item.email}")

        @unit_of_work()
        def tx_fn(
            tx: ManagedTransaction, item: schemas.CreateStudent
        ) -> models.Student:
            self.__logger.debug("Hashing student password")
            password_hash = self.__auth_service.hash_password(item.password)
            item.password = password_hash

            query = (
                "CREATE (s:$student_node_name {{name: $name, email: $email, password: $password, enabled: $enabled}}) "
                "RETURN s"
            )

            self.__logger.debug(f"Executing student creation query for {item.email}")
            result = tx.run(
                query,
                item.model_dump(by_alias=True),
                student_node_name=self.__student_node_name,
            )
            node = result.single(strict=True)
            self.__logger.debug("Student node created in Neo4j")
            return models.Student(**node["s"])

        self.__logger.debug(
            f"Checking if student with email {item.email} already exists"
        )
        user_by_email = self.get_student_by_email(item.email)
        if user_by_email:
            self.__logger.warning(
                f"Student creation failed: email {item.email} already exists"
            )
            raise ValueError(f"Student with email {item.email} already exists")

        new_user = self.__session.execute_write(tx_fn, item)
        self.__logger.info(f"Student created successfully: {item.email}")
        return new_user

    def get_student(self, id: str) -> models.Student | None:
        @unit_of_work()
        def tx_fn(tx: ManagedTransaction, id: str) -> models.Student | None:
            query = "MATCH (s:$student_node_name) WHERE id(s) = $id RETURN s"
            result = tx.run(query, id=id, student_node_name=self.__student_node_name)
            record = result.single()
            if record:
                return models.Student(**record["s"])
            return None

        return self.__session.execute_read(tx_fn, id)

    def get_student_by_email(self, email: str) -> models.Student | None:
        @unit_of_work()
        def tx_fn(tx: ManagedTransaction, email: str) -> models.Student | None:
            query = "MATCH (s:$student_node_name) WHERE s.email = $email RETURN s"
            result = tx.run(
                query,
                email=email,
                student_node_name=self.__student_node_name,
            )
            record = result.single()
            if record:
                return models.Student(**record["s"])
            return None

        return self.__session.execute_read(tx_fn, email)

    def update_student(
        self,
        id: str,
        *,
        to_update: schemas.UpdateStudent,
    ) -> models.Student:
        @unit_of_work()
        def tx_fn(
            tx: ManagedTransaction, id: str, to_update: schemas.UpdateStudent
        ) -> models.Student:
            params = to_update.model_dump(exclude_unset=True, by_alias=True)
            query = """
            MATCH (s:$student_node_name) WHERE id(s) = $id
            SET s += $props
            RETURN s
            """
            result = tx.run(
                query, id=id, props=params, student_node_name=self.__student_node_name
            )
            node = result.single(strict=True)
            return models.Student(**node["s"])

        return self.__session.execute_write(tx_fn, id, to_update)

    def get_student_trajectory(
        self,
        student_id: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
        timestamp_order: Literal["ASC", "DESC"] = "DESC",
    ) -> list[models.StudentTrajectory]:
        @unit_of_work()
        def tx_fn(
            tx: ManagedTransaction,
            student_id: str,
            *,
            limit: int | None = None,
            offset: int | None = None,
            timestamp_order: Literal["ASC", "DESC"] = "DESC",
        ) -> list[models.StudentTrajectory]:
            if limit is not None and limit <= 0:
                raise ValueError("Limit must be a positive integer")
            if offset is not None and offset < 0:
                raise ValueError("Offset must be a non-negative integer")

            query = f"""
            MATCH (s:{self.__student_node_name})-[:{self.__trajectory_rel_name}]->(t:{self.__trajectory_node_name})
            WHERE id(s) = $student_id
            """
            params: dict[str, Any] = {}
            if limit is not None:
                query += " LIMIT $limit"
                params["limit"] = limit
            if offset is not None:
                query += " SKIP $offset"
                params["offset"] = offset
            query += (
                f" RETURN t, id(s) AS student_id ORDER BY t.timestamp {timestamp_order}"
            )

            result = tx.run(
                query,
                params,
                student_id=student_id,
            )
            trajectories = []
            for record in result:
                data = dict(record["t"])
                data["student_id"] = record["student_id"]
                trajectories.append(models.StudentTrajectory(**data))
            return trajectories

        return self.__session.execute_read(
            tx_fn,
            student_id,
            limit=limit,
            offset=offset,
            timestamp_order=timestamp_order,
        )

    def get_student_trajectory_by_query_exact_match(
        self,
        student_id: str,
        query: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[models.StudentTrajectory]:
        @unit_of_work()
        def tx_fn(
            tx: ManagedTransaction,
            student_id: str,
            query: str,
            limit: int | None = None,
            offset: int | None = None,
        ) -> list[models.StudentTrajectory]:

            statement = f"""
            MATCH (s:{self.__student_node_name})-[:{self.__trajectory_rel_name}]->(t:{self.__trajectory_node_name})
            WHERE id(s) = $student_id AND t.{self.__trajectory_full_text_index_field} = $query_hash
            """

            parameters = {}
            if limit is not None:
                statement += " LIMIT $limit"
                parameters["limit"] = limit
            if offset is not None:
                statement += " SKIP $offset"
                parameters["offset"] = offset
            statement += " RETURN t, id(s) AS student_id ORDER BY t.timestamp DESC"

            result = tx.run(
                statement,
                parameters,
                student_id=student_id,
                query_hash=hash_string(query),
            )
            trajectories = []
            for record in result:
                data = dict(record["t"])
                data["student_id"] = record["student_id"]
                trajectories.append(models.StudentTrajectory(**data))
            return trajectories

        return self.__session.execute_read(
            tx_fn, student_id, query, limit=limit, offset=offset
        )

    def get_student_trajectory_by_query_similarity(
        self,
        student_id: str,
        query: str,
        *,
        threshold: float | None = None,
        limit: int | None = None,
    ) -> list[models.StudentTrajectory]:
        config = {}
        if limit is not None:
            config["top_k"] = limit
        if threshold is not None:
            config["similarity_threshold"] = threshold
        try:
            result = self.__rag.retriever.get_search_results(
                query_text=query,
                retriever_config=config,
                filters={"student_id": student_id},
            )
            return [models.StudentTrajectory(**item.data()) for item in result.records]
        except Exception as e:
            self.__logger.warning(
                f"Vector retriever search failed during similarity retrieval: {e}"
            )
            return []

    def add_trajectory_entry(
        self,
        student_id: str,
        trajectory_entry: models.StudentTrajectory,
    ) -> models.StudentTrajectory:
        self.__logger.info(f"Adding trajectory entry for student {student_id}")
        self.__logger.debug(f"Query: {trajectory_entry.query[:100]}...")

        @unit_of_work()
        def tx_fn(
            tx: ManagedTransaction,
            student_id: str,
            trajectory_entry: models.StudentTrajectory,
        ) -> models.StudentTrajectory:
            query = f"""
            MATCH (s:{self.__student_node_name}) WHERE id(s) = $student_id
            CREATE (t:{self.__trajectory_node_name} $props)
            CREATE (s)-[rel:{self.__trajectory_rel_name}]->(t)
            WITH s, t, rel
            OPTIONAL MATCH (s)-[prev_rel:{self.__trajectory_rel_name}]->(prev_t:{self.__trajectory_node_name})
            WHERE prev_t <> t
            ORDER BY prev_t.timestamp DESC
            LIMIT 1
            CREATE (t)-[:{self.__trajectory_prev_rel_name}]->(prev_t)
            RETURN t, id(s) AS student_id
            """

            params = trajectory_entry.model_dump(by_alias=True)
            self.__logger.debug("Hashing query for full-text index")
            params[self.__trajectory_full_text_index_field] = hash_string(
                trajectory_entry.query
            )
            self.__logger.debug("Embedding query for vector search")
            params[self.__trajectory_vector_index_field] = (
                self.__embedder.embed_documents_sync([trajectory_entry.query])[0]
            )

            self.__logger.debug("Creating trajectory node in Neo4j")
            result = tx.run(
                query,
                params,
                student_id=student_id,
            )
            node = result.single(strict=True)
            data = dict(node["t"])
            data["student_id"] = node["student_id"]
            self.__logger.debug("Trajectory node created successfully")
            return models.StudentTrajectory(**data)

        result = self.__session.execute_write(tx_fn, student_id, trajectory_entry)
        self.__logger.info(
            f"Trajectory entry added successfully for student {student_id}"
        )
        return result

    def increment_trajectory_query_repeat_count(
        self, trajectory_id: str, *, increment: int = 1
    ) -> models.StudentTrajectory:
        self.__logger.debug(
            f"Incrementing query repeat count for trajectory {trajectory_id} by {increment}"
        )

        @unit_of_work()
        def tx_fn(
            tx: ManagedTransaction,
            trajectory_id: str,
            increment: int,
        ) -> models.StudentTrajectory:
            query = f"""
            MATCH (t:{self.__trajectory_node_name}) WHERE id(t) = $trajectory_id
            SET t.query_repeat_count = t.query_repeat_count + $increment
            RETURN t
            """
            result = tx.run(
                query,
                trajectory_id=trajectory_id,
                increment=increment,
            )
            node = result.single(strict=True)
            return models.StudentTrajectory(**node["t"])

        return self.__session.execute_write(tx_fn, trajectory_id, increment=increment)

    def delete_student(self, id: str) -> None:
        to_update = schemas.UpdateStudent(enabled=False)
        self.update_student(id, to_update=to_update)


# TODO: finish the fixes

from contextlib import contextmanager
import logging
from neo4j import Driver, GraphDatabase

logger = logging.getLogger(__name__)


class Neo4jSession:
    def __init__(self, driver: Driver, **session_kwargs):
        self.__driver = driver
        self.__session_kwargs = session_kwargs

    def __enter__(self):
        logger.info("Opening Neo4j session")
        self.__session = self.__driver.session(**self.__session_kwargs)
        return self.__session

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Closing Neo4j session")
        self.__session.close()


class Neo4jGateway:
    def __init__(self, uri: str, *, user: str, password: str):
        self.__uri = uri
        self.__user = user
        self.__password = password

    def __enter__(self):
        auth = (
            (self.__user, self.__password) if self.__user and self.__password else None
        )
        self.__driver = GraphDatabase.driver(self.__uri, auth=auth)
        logger.info(f"Connected to Neo4j at {self.__uri} with user {self.__user}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__driver.close()
        logger.info("Neo4j driver closed")

    @contextmanager
    def session(self, **session_kwargs):
        with Neo4jSession(self.__driver, **session_kwargs) as session:
            yield session

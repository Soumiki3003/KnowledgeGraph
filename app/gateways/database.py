from contextlib import contextmanager
import logging
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import orm

logger = logging.getLogger(__name__)


class DatabaseGateway:
    def __init__(self, uri: str):
        self.__engine = create_engine(uri)
        self._session_factory = orm.scoped_session(
            orm.sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.__engine,
            ),
        )

    def create_database(self):
        SQLModel.metadata.create_all(self.__engine)

    @contextmanager
    def session(self):
        session: Session = self._session_factory()
        try:
            yield session
        except Exception:
            logger.exception("Session rollback because of exception")
            session.rollback()
            raise
        finally:
            session.close()

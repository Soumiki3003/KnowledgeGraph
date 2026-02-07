from app import gateways, models
from sqlmodel import select

from app.utils import hash_string
from .auth import AuthService


class StudentService:
    def __init__(
        self,
        db_gateway: gateways.DatabaseGateway,
        auth_service: AuthService,
    ):
        self.__db_gateway = db_gateway
        self.__auth_service = auth_service

    def create(self, item: models.Student) -> models.Student:
        password_hash = self.__auth_service.hash_password(item.password)
        new_entity = models.Student(
            name=item.name, email=item.email, password=password_hash
        )
        with self.__db_gateway.session() as session:
            session.add(new_entity)
            session.commit()
            session.refresh(new_entity)
        return new_entity

    def get(self, id: int) -> models.Student | None:
        statement = select(models.Student).where(models.Student.id == id)
        with self.__db_gateway.session() as session:
            return session.exec(statement).first()

    def get_by_email(self, email: str) -> models.Student | None:
        statement = select(models.Student).where(models.Student.email == email)
        with self.__db_gateway.session() as session:
            return session.exec(statement).first()

    def update(
        self,
        id: int,
        *,
        name: str | None = None,
        email: str | None = None,
        password: str | None = None,
        enabled: bool | None = None,
    ) -> models.Student:
        with self.__db_gateway.session() as session:
            student = session.get(models.Student, id)
            if not student:
                raise ValueError(f"Student with id {id} not found")

            if name is not None:
                student.name = name
            if email is not None:
                student.email = email
            if password is not None:
                student.password = self.__auth_service.hash_password(password)
            if enabled is not None:
                student.enabled = enabled

            session.add(student)
            session.commit()
            session.refresh(student)
        return student


class StudentTrajectoryService:
    def __init__(self, db_gateway: gateways.DatabaseGateway):
        self.__db_gateway = db_gateway

    def create(self, item: models.StudentTrajectory) -> models.StudentTrajectory:
        with self.__db_gateway.session() as session:
            session.add(item)
            session.commit()
            session.refresh(item)
        return item

    def get_by_query(
        self,
        student_id: int,
        query: str,
    ) -> list[models.StudentTrajectory]:
        query_hash = hash_string(query)

        statement = select(models.StudentTrajectory).where(
            models.StudentTrajectory.student_id == student_id,
            models.StudentTrajectory.query_hash == query_hash,
        )
        with self.__db_gateway.session() as session:
            return session.exec(statement).all()

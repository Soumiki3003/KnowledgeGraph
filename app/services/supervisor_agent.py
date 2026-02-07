import logging
import time
from .student import StudentService

logger = logging.getLogger(__name__)


class SupervisorAgentService:
    def __init__(self, student_service: StudentService):
        self.__student_service = student_service

    def retrieve_context(self, student_id: int, query: str):
        logger.info(f"\nRetrieving context for: '{query}' (student={student_id})")

        start_time = time.time()
        # TODO: implement supervisor_agent.py

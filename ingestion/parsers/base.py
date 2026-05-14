#abstract parser interface

from abc import ABC, abstractmethod
from ingestion.schema import RawDocument

class BaseParser(ABC):
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        ...

    @abstractmethod
    def parse(self, file_path: str) -> RawDocument:
        ...
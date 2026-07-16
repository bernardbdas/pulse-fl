from sqlmodel import SQLModel, create_engine, Session
from pulse_fl.config import settings

class DatabaseConnectionManager:
    """
    Singleton Database Connection Manager.
    Coordinates SQLite or PostgreSQL engines, pooling, and session creation.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        db_url = settings.DATABASE_URL
        print(f"[DATABASE] Initializing connection manager for: {db_url}")
        
        # Check if database is SQLite
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
            
        self.engine = create_engine(db_url, connect_args=connect_args)
        self._initialized = True

    def init_db(self):
        """Creates database schema tables if they do not exist."""
        SQLModel.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Creates and returns a new SQLModel Session."""
        return Session(self.engine)

def get_session_dependency():
    """FastAPI dependency generator yielding sessions."""
    manager = DatabaseConnectionManager()
    with manager.get_session() as session:
        yield session

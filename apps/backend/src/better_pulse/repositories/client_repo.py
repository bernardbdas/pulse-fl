from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import Session, select
from better_pulse.schemas.db_models import Client

class ClientRepository:
    """
    Repository Pattern: Encapsulates Client database actions.
    """
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, device_id: str) -> Optional[Client]:
        return self.session.get(Client, device_id)

    def get_all(self) -> List[Client]:
        return list(self.session.exec(select(Client)).all())

    def create(self, device_id: str, device_model: Optional[str] = None, emergency_email: Optional[str] = None) -> Client:
        client = Client(device_id=device_id, device_model=device_model, emergency_email=emergency_email)
        self.session.add(client)
        self.session.commit()
        self.session.refresh(client)
        return client

    def update_active(self, client: Client):
        client.last_active = datetime.now(timezone.utc)
        self.session.add(client)
        self.session.commit()

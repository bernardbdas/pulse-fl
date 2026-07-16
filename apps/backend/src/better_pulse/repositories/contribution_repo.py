from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import Session, select
from better_pulse.schemas.db_models import ClientContribution

class ContributionRepository:
    """
    Repository Pattern: Encapsulates Client local training update database actions.
    """
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        client_id: str,
        round_number: int,
        local_loss: float,
        local_accuracy: Optional[float],
        sample_count: int,
        update_file_path: str
    ) -> ClientContribution:
        contribution = ClientContribution(
            client_id=client_id,
            round_number=round_number,
            local_loss=local_loss,
            local_accuracy=local_accuracy,
            sample_count=sample_count,
            update_file_path=update_file_path,
            uploaded_at=datetime.now(timezone.utc)
        )
        self.session.add(contribution)
        self.session.commit()
        self.session.refresh(contribution)
        return contribution

    def get_by_round(self, round_number: int) -> List[ClientContribution]:
        statement = select(ClientContribution).where(ClientContribution.round_number == round_number)
        return list(self.session.exec(statement).all())

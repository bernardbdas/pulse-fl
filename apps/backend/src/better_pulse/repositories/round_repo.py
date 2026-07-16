from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import Session, select
from better_pulse.schemas.db_models import FLRound, GlobalModelHistory

class RoundRepository:
    """
    Repository Pattern: Encapsulates Federated Learning Rounds and History database actions.
    """
    def __init__(self, session: Session):
        self.session = session

    def get_active_round(self) -> Optional[FLRound]:
        statement = select(FLRound).where(FLRound.status == "OPEN")
        return self.session.exec(statement).first()

    def get_round_by_number(self, round_number: int) -> Optional[FLRound]:
        return self.session.get(FLRound, round_number)

    def get_all_rounds(self) -> List[FLRound]:
        return list(self.session.exec(select(FLRound)).all())

    def start_new_round(self, round_number: int, global_model_path: str) -> FLRound:
        # Close any existing open rounds
        statement = select(FLRound).where(FLRound.status == "OPEN")
        open_rounds = self.session.exec(statement).all()
        for r in open_rounds:
            r.status = "COMPLETED"
            if not r.end_time:
                r.end_time = datetime.now(timezone.utc)
            self.session.add(r)
            
        new_round = FLRound(
            round_number=round_number,
            status="OPEN",
            global_model_path=global_model_path,
            start_time=datetime.now(timezone.utc)
        )
        self.session.add(new_round)
        self.session.commit()
        self.session.refresh(new_round)
        return new_round

    def log_global_model_history(
        self, 
        round_number: int, 
        weights_path: str, 
        loss: Optional[float] = None, 
        accuracy: Optional[float] = None
    ) -> GlobalModelHistory:
        history = GlobalModelHistory(
            round_number=round_number,
            weights_path=weights_path,
            loss=loss,
            accuracy=accuracy,
            updated_at=datetime.now(timezone.utc)
        )
        self.session.add(history)
        self.session.commit()
        self.session.refresh(history)
        return history

    def get_global_history(self) -> List[GlobalModelHistory]:
        return list(self.session.exec(select(GlobalModelHistory).order_by(GlobalModelHistory.round_number)).all())

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import Session, select
from pulse_fl.schemas.db_models import SignalSession, AnomalyAlert

class AlertRepository:
    """
    Repository Pattern: Encapsulates database actions for SignalSessions and AnomalyAlerts.
    """
    def __init__(self, session: Session):
        self.session = session

    # Signal Sessions
    def create_session(self, client_id: str) -> SignalSession:
        db_session = SignalSession(client_id=client_id, is_active=True)
        self.session.add(db_session)
        self.session.commit()
        self.session.refresh(db_session)
        return db_session

    def get_active_sessions(self) -> List[SignalSession]:
        statement = select(SignalSession).where(SignalSession.is_active == True)
        return list(self.session.exec(statement).all())

    def close_session(self, session_id: int):
        db_session = self.session.get(SignalSession, session_id)
        if db_session:
            db_session.is_active = False
            db_session.ended_at = datetime.now(timezone.utc)
            self.session.add(db_session)
            self.session.commit()

    # Anomaly Alerts
    def create_alert(
        self, 
        client_id: str, 
        session_id: Optional[int], 
        confidence: float, 
        heart_rate: int,
        activity_state: str = "STATIONARY",
        is_suppressed: bool = False
    ) -> AnomalyAlert:
        alert = AnomalyAlert(
            client_id=client_id,
            session_id=session_id,
            confidence=confidence,
            heart_rate=heart_rate,
            activity_state=activity_state,
            is_suppressed=is_suppressed
        )
        self.session.add(alert)
        self.session.commit()
        self.session.refresh(alert)
        return alert

    def update_alert_gemma_report(self, alert_id: int, report: str):
        alert = self.session.get(AnomalyAlert, alert_id)
        if alert:
            alert.gemma_report = report
            self.session.add(alert)
            self.session.commit()

    def get_all_alerts(self) -> List[AnomalyAlert]:
        statement = select(AnomalyAlert).order_by(AnomalyAlert.timestamp.desc())
        return list(self.session.exec(statement).all())

from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field

class Client(SQLModel, table=True):
    device_id: str = Field(primary_key=True, index=True)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    device_model: Optional[str] = Field(default=None)
    emergency_email: Optional[str] = Field(default=None)

class FLRound(SQLModel, table=True):
    round_number: int = Field(primary_key=True)
    status: str = Field(default="OPEN")  # "OPEN", "AGGREGATING", "COMPLETED", "FAILED"
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = Field(default=None)
    global_model_path: Optional[str] = Field(default=None)

class ClientContribution(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: str = Field(foreign_key="client.device_id", index=True)
    round_number: int = Field(foreign_key="flround.round_number", index=True)
    local_loss: float = Field(default=0.0)
    local_accuracy: Optional[float] = Field(default=None)
    sample_count: int = Field(default=0)
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    update_file_path: str

class GlobalModelHistory(SQLModel, table=True):
    round_number: int = Field(primary_key=True)
    accuracy: Optional[float] = Field(default=None)
    loss: Optional[float] = Field(default=None)
    weights_path: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SignalSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: str = Field(foreign_key="client.device_id", index=True)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = Field(default=None)
    is_active: bool = Field(default=True)

class AnomalyAlert(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: str = Field(foreign_key="client.device_id", index=True)
    session_id: Optional[int] = Field(default=None, foreign_key="signalsession.id")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(default=0.0)
    heart_rate: int = Field(default=75)
    activity_state: str = Field(default="STATIONARY")
    is_suppressed: bool = Field(default=False)
    gemma_report: Optional[str] = Field(default=None)

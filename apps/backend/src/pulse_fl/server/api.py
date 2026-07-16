import os
import shutil
import torch
import asyncio
import httpx
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from safetensors.torch import save_file, load_file

from pulse_fl.database import get_session_dependency, DatabaseConnectionManager
from pulse_fl.config import settings
from pulse_fl.schemas.db_models import Client, FLRound, ClientContribution, GlobalModelHistory, SignalSession, AnomalyAlert
from pulse_fl.repositories import ClientRepository, RoundRepository, ContributionRepository, AlertRepository
from pulse_fl.aggregation.strategy import FedAvgStrategy
from pulse_fl.server.websocket_manager import websocket_manager

router = APIRouter()

def aggregate_round_background(round_number: int):
    """
    Background worker task to aggregate model weights from the completed round,
    log the training metrics, export the model to ExecuTorch (.pte) format, 
    and open the next round. Uses Strategy and Repository patterns.
    """
    print(f"[BACKGROUND] Starting aggregation for round {round_number}...")
    
    db_manager = DatabaseConnectionManager()
    with db_manager.get_session() as session:
        round_repo = RoundRepository(session)
        contrib_repo = ContributionRepository(session)
        
        r = round_repo.get_round_by_number(round_number)
        if not r or r.status != "AGGREGATING":
            print(f"[BACKGROUND] Round {round_number} is not in AGGREGATING state. Aborting.")
            return

        try:
            contributions = contrib_repo.get_by_round(round_number)
            if len(contributions) < settings.MIN_CLIENTS_FOR_AGGREGATION:
                raise ValueError(f"Not enough contributions. Expected {settings.MIN_CLIENTS_FOR_AGGREGATION}, got {len(contributions)}")

            # 1. Aggregate state dicts using the Strategy Pattern
            strategy = FedAvgStrategy()
            aggregated_weights = strategy.aggregate(contributions)

            # 2. Save aggregated weights
            next_round_number = round_number + 1
            next_model_path = settings.GLOBAL_MODELS_DIR / f"global_model_round_{next_round_number}.safetensors"
            save_file(aggregated_weights, str(next_model_path))

            # 3. Export to ExecuTorch format (.pte) for the next round
            pte_path = settings.GLOBAL_MODELS_DIR / f"global_model_round_{next_round_number}.pte"
            try:
                from pulse_fl.models.factory import ModelFactory
                model = ModelFactory.create_model("ECGNet")
                model.load_state_dict(aggregated_weights)
                model.eval()

                example_input = torch.randn(1, settings.ECG_INPUT_CHANNELS, settings.ECG_SEQUENCE_LENGTH)
                exported_program = torch.export.export(model, (example_input,))
                
                try:
                    from executorch.exir import to_edge
                    edge_program = to_edge(exported_program)
                    et_program = edge_program.to_executorch()
                    with open(pte_path, "wb") as f:
                        f.write(et_program.buffer)
                    print(f"[BACKGROUND] Aggregated model exported to ExecuTorch at {pte_path}")
                except ImportError:
                    torch.save(exported_program, pte_path)
                    print(f"[BACKGROUND] Exported fallback traced model to {pte_path}")
            except Exception as ex_err:
                print(f"[BACKGROUND] ExecuTorch model compile failed (using fallback): {ex_err}")

            # 4. Calculate stats (weighted loss & accuracy)
            total_samples = sum(c.sample_count for c in contributions)
            weighted_loss = sum(c.local_loss * (c.sample_count / total_samples) for c in contributions)
            weighted_accuracy = sum((c.local_accuracy or 0.0) * (c.sample_count / total_samples) for c in contributions)

            # 5. Log history and transition rounds
            round_repo.log_global_model_history(round_number, weights_path=str(next_model_path), loss=weighted_loss, accuracy=weighted_accuracy)
            round_repo.start_new_round(next_round_number, global_model_path=str(next_model_path))
            
            print(f"[BACKGROUND] Round {round_number} successfully aggregated. Round {next_round_number} is now OPEN.")

        except Exception as err:
            print(f"[BACKGROUND] Aggregation failed for round {round_number}: {err}")
            r.status = "FAILED"
            session.add(r)
            session.commit()


@router.post("/clients/register")
def register_client(
    device_id: str = Form(...),
    device_model: Optional[str] = Form(None),
    emergency_email: Optional[str] = Form(None),
    session: Session = Depends(get_session_dependency)
):
    """Registers a client device or updates its last active status and emergency contact email."""
    repo = ClientRepository(session)
    client = repo.get_by_id(device_id)
    if not client:
        client = repo.create(device_id, device_model, emergency_email)
        return {"status": "registered", "device_id": client.device_id, "message": "New device registered."}
    
    if emergency_email:
        client.emergency_email = emergency_email
        session.add(client)
        session.commit()
    repo.update_active(client)
    return {"status": "active", "device_id": client.device_id, "message": "Device activity timestamp updated."}


@router.get("/rounds/active")
def get_active_round_metadata(session: Session = Depends(get_session_dependency)):
    """Retrieves current active round details and model compilation formats available."""
    round_repo = RoundRepository(session)
    contrib_repo = ContributionRepository(session)
    
    active_round = round_repo.get_active_round()
    if not active_round:
        raise HTTPException(status_code=404, detail="No active federated learning round found.")
    
    contributions = contrib_repo.get_by_round(active_round.round_number)
    
    round_num = active_round.round_number
    safetensors_exists = (settings.GLOBAL_MODELS_DIR / f"global_model_round_{round_num}.safetensors").exists()
    pte_exists = (settings.GLOBAL_MODELS_DIR / f"global_model_round_{round_num}.pte").exists()

    return {
        "round_number": active_round.round_number,
        "status": active_round.status,
        "start_time": active_round.start_time,
        "participants_count": len(contributions),
        "min_participants_required": settings.MIN_CLIENTS_FOR_AGGREGATION,
        "available_formats": {
            "safetensors": safetensors_exists,
            "executorch_pte": pte_exists
        }
    }


@router.get("/rounds/download")
def download_model(
    format: str = "pte",
    session: Session = Depends(get_session_dependency)
):
    """Downloads the global model parameters for the active round in `.pte` or `.safetensors` format."""
    round_repo = RoundRepository(session)
    active_round = round_repo.get_active_round()
    if not active_round:
        raise HTTPException(status_code=404, detail="No active federated learning round found.")
        
    round_num = active_round.round_number
    
    if format == "safetensors":
        file_path = settings.GLOBAL_MODELS_DIR / f"global_model_round_{round_num}.safetensors"
        media_type = "application/octet-stream"
    else:
        file_path = settings.GLOBAL_MODELS_DIR / f"global_model_round_{round_num}.pte"
        media_type = "application/octet-stream"
        
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file in format '{format}' not found for round {round_num}.")
        
    return FileResponse(path=str(file_path), filename=file_path.name, media_type=media_type)


@router.post("/rounds/upload")
def upload_client_contribution(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...),
    round_number: int = Form(...),
    local_loss: float = Form(...),
    local_accuracy: float = Form(...),
    sample_count: int = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session_dependency)
):
    """
    Receives updated model parameters from a client.
    Stores files securely and triggers aggregation if participation limits are reached.
    """
    client_repo = ClientRepository(session)
    round_repo = RoundRepository(session)
    contrib_repo = ContributionRepository(session)
    
    client = client_repo.get_by_id(client_id)
    if not client:
        raise HTTPException(status_code=403, detail="Client is not registered. Please register first.")

    active_round = round_repo.get_active_round()
    if not active_round:
        raise HTTPException(status_code=400, detail="No active federated learning round.")
        
    if active_round.round_number != round_number:
        raise HTTPException(
            status_code=400, 
            detail=f"Mismatched round. Server active round is {active_round.round_number}, received contribution for {round_number}."
        )

    existing_contributions = contrib_repo.get_by_round(round_number)
    if any(c.client_id == client_id for c in existing_contributions):
        raise HTTPException(status_code=409, detail=f"Client '{client_id}' already uploaded updates for round {round_number}.")

    dest_path = settings.CLIENT_UPDATES_DIR / f"round_{round_number}_client_{client_id}.safetensors"
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded model file: {e}")

    contrib_repo.create(
        client_id=client_id,
        round_number=round_number,
        local_loss=local_loss,
        local_accuracy=local_accuracy,
        sample_count=sample_count,
        update_file_path=str(dest_path)
    )
    
    client_repo.update_active(client)

    updated_contributions = contrib_repo.get_by_round(round_number)
    if len(updated_contributions) >= settings.MIN_CLIENTS_FOR_AGGREGATION:
        active_round.status = "AGGREGATING"
        session.add(active_round)
        session.commit()
        
        background_tasks.add_task(aggregate_round_background, round_number)
        return {"status": "aggregating", "message": f"Threshold reached. Aggregation for round {round_number} started."}

    return {"status": "accepted", "message": f"Contribution accepted. {len(updated_contributions)}/{settings.MIN_CLIENTS_FOR_AGGREGATION} uploads received."}


@router.get("/status")
def get_system_status(session: Session = Depends(get_session_dependency)):
    """API for backend analytics and learning progress monitoring."""
    round_repo = RoundRepository(session)
    client_repo = ClientRepository(session)
    alert_repo = AlertRepository(session)
    
    rounds = round_repo.get_all_rounds()
    history = round_repo.get_global_history()
    clients = client_repo.get_all()
    active_sessions = alert_repo.get_active_sessions()
    
    return {
        "active_clients_count": len(clients),
        "total_rounds_processed": len(rounds),
        "active_streams_count": len(active_sessions),
        "active_streams": [{"client_id": s.client_id, "session_id": s.id, "started_at": s.started_at} for s in active_sessions],
        "metrics_history": [
            {
                "round_number": h.round_number,
                "loss": h.loss,
                "accuracy": h.accuracy,
                "updated_at": h.updated_at
            }
            for h in history
        ]
    }


async def generate_gemma_clinical_report(alert_id: int):
    """Asynchronous background worker to query local Ollama running Gemma-4 for clinical analysis, and dispatch email alerts."""
    print(f"[GEMMA] Generating clinical analysis for Alert ID {alert_id}...")
    db_manager = DatabaseConnectionManager()
    
    with db_manager.get_session() as session:
        alert_repo = AlertRepository(session)
        client_repo = ClientRepository(session)
        alert = session.get(AnomalyAlert, alert_id)
        if not alert:
            return
            
        prompt = f"""
You are a cardiological clinical assistant AI. Analyze the following wearable ECG telemetry anomaly report:
- Client Device: {alert.client_id}
- Detected Anomaly: Cardiac Arrhythmia (Atrial Fibrillation / AFib)
- Model Prediction Confidence: {alert.confidence * 100:.1f}%
- Estimated Heart Rate: {alert.heart_rate} BPM
- Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Provide a concise clinical report summary for a physician. Include:
1. Potential physiological risks associated with this reading (atrial stasis, stroke risk, tachycardia).
2. Suggested immediate triage actions or recommendation for the patient (e.g., rest, check symptoms, seek emergency care if chest pain occurs).

Keep the tone professional, objective, and medically precise. Limit the response to 3-4 bullet points. Do not include introductory or concluding conversational filler.
"""
        gemma_report_text = ""
        try:
            url = f"{settings.OLLAMA_API_URL}/api/generate"
            payload = {
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    gemma_report_text = result.get("response", "").strip()
                    alert_repo.update_alert_gemma_report(alert.id, gemma_report_text)
                    print(f"[GEMMA] Clinical analysis completed for Alert ID {alert_id}.")
                else:
                    print(f"[GEMMA] Ollama responded with status code {response.status_code}")
        except Exception as e:
            print(f"[GEMMA] Ollama query failed for Alert ID {alert_id}: {e}")

        # Send emergency email alert to registered contact
        client_record = client_repo.get_by_id(alert.client_id)
        if client_record and client_record.emergency_email:
            from pulse_fl.services.notification_service import NotificationService
            NotificationService.send_emergency_email(
                recipient_email=client_record.emergency_email,
                client_id=alert.client_id,
                heart_rate=alert.heart_rate,
                confidence=alert.confidence,
                activity_state=alert.activity_state,
                gemma_report=gemma_report_text
            )


@router.websocket("/signals/stream/{client_id}")
async def stream_signals(websocket: WebSocket, client_id: str):
    """WebSocket endpoint to receive and process real-time ECG signal data points from wearables."""
    db_manager = DatabaseConnectionManager()
    session = db_manager.get_session()
    
    client_repo = ClientRepository(session)
    round_repo = RoundRepository(session)
    alert_repo = AlertRepository(session)
    
    try:
        client = client_repo.get_by_id(client_id)
        if not client:
            client = client_repo.create(client_id, "StreamingWearableNode")
            
        db_session = alert_repo.create_session(client_id)
        await websocket_manager.connect(client_id, websocket)
        print(f"[WEBSOCKET] Client '{client_id}' connected. Session ID: {db_session.id}")
        
        buffer = []
        window_size = settings.ECG_SEQUENCE_LENGTH  # 1000
        current_activity = "STATIONARY"
        
        active_round = round_repo.get_active_round()
        round_num = active_round.round_number if active_round else 0
        model_path = settings.GLOBAL_MODELS_DIR / f"global_model_round_{round_num}.safetensors"
        
        model = None
        if model_path.exists():
            try:
                from pulse_fl.models.factory import ModelFactory
                model = ModelFactory.create_model("ECGNet")
                model.load_state_dict(load_file(str(model_path)))
                model.eval()
                print(f"[WEBSOCKET] Model loaded from {model_path.name} for streaming inference.")
            except Exception as e:
                print(f"[WEBSOCKET] Model load error: {e}")
        
        while True:
            data = await websocket.receive_json()
            values = data.get("values", [])
            current_activity = data.get("activity_state", current_activity).upper()
            buffer.extend(values)
            
            if len(buffer) > 2000:
                buffer = buffer[-2000:]
                
            if len(buffer) >= window_size and model is not None:
                window = buffer[-window_size:]
                input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1).squeeze(0)
                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_class].item()
                
                # Developer test override to verify suppression and alert notification pipelines
                if client_id == "device_watch_test" and len(buffer) >= 1500:
                    pred_class = 1
                    confidence = 0.95
                
                if pred_class == 1 and confidence >= 0.70:
                    # Suppress alerts if user is walking, running, or exercising to avoid false positives
                    is_suppressed = current_activity in ["WALKING", "RUNNING", "EXERCISING"]
                    
                    alert = alert_repo.create_alert(
                        client_id=client_id,
                        session_id=db_session.id,
                        confidence=confidence,
                        heart_rate=115,
                        activity_state=current_activity,
                        is_suppressed=is_suppressed
                    )
                    
                    if is_suppressed:
                        print(f"[WEBSOCKET] Cardiac anomaly suppressed for client {client_id} due to activity state: {current_activity}")
                        await websocket.send_json({
                            "type": "STATUS",
                            "message": f"Elevated metrics (Suppressed - Activity: {current_activity})",
                            "confidence": confidence,
                            "heart_rate": 115
                        })
                    else:
                        print(f"[WEBSOCKET] Cardiac anomaly (arrhythmia) detected for client {client_id}! ID: {alert.id}")
                        
                        await websocket.send_json({
                            "type": "ALERT",
                            "message": f"Cardiac Arrhythmia Detected! Confidence: {confidence * 100:.1f}%",
                            "alert_id": alert.id,
                            "confidence": confidence,
                            "heart_rate": 115
                        })
                        
                        asyncio.create_task(generate_gemma_clinical_report(alert.id))
                else:
                    await websocket.send_json({
                        "type": "STATUS",
                        "message": "Sinus Rhythm",
                        "confidence": confidence,
                        "heart_rate": 70
                    })
            else:
                await websocket.send_json({
                    "type": "ACK",
                    "message": f"Buffering ({len(buffer)}/{window_size} values)",
                    "heart_rate": 70
                })
    except WebSocketDisconnect:
        print(f"[WEBSOCKET] Client '{client_id}' disconnected.")
    except Exception as err:
        print(f"[WEBSOCKET] Error in signal stream: {err}")
    finally:
        websocket_manager.disconnect(client_id)
        try:
            alert_repo.close_session(db_session.id)
        except Exception:
            pass
        session.close()


@router.get("/alerts")
def get_anomaly_alerts(session: Session = Depends(get_session_dependency)):
    """Retrieves all logged wearable anomaly alerts."""
    repo = AlertRepository(session)
    alerts = repo.get_all_alerts()
    return [
        {
            "id": a.id,
            "client_id": a.client_id,
            "session_id": a.session_id,
            "timestamp": a.timestamp,
            "confidence": a.confidence,
            "heart_rate": a.heart_rate,
            "activity_state": a.activity_state,
            "is_suppressed": a.is_suppressed,
            "gemma_report": a.gemma_report
        }
        for a in alerts
    ]

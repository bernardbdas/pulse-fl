import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from better_pulse.config import settings

class NotificationService:
    """
    Service to dispatch emergency email alerts to registered contacts.
    If the SMTP connection fails, logs the notification to storage/sent_emails.log.
    """
    @staticmethod
    def send_emergency_email(
        recipient_email: str,
        client_id: str,
        heart_rate: int,
        confidence: float,
        activity_state: str,
        gemma_report: str
    ) -> bool:
        subject = f"[URGENT] Better-Pulse Cardiac Alert - Arrhythmia Detected for Device {client_id}"
        
        body = f"""Better-Pulse Emergency Cardiac Telemetry Notification

Dear Emergency Contact,

This is an automated cardiac anomaly alert from the Better-Pulse wearable monitoring system.

Telemetry Details:
- Device ID: {client_id}
- Detected Anomaly: Cardiac Arrhythmia (Atrial Fibrillation / AFib)
- Heart Rate Estimate: {heart_rate} BPM
- Classification Confidence: {confidence * 100:.1f}%
- Wearable Activity State: {activity_state}

---------------------------------------------------------
Ollama Gemma-4 Assisted Clinical Report & Triage Advice:
{gemma_report or "Gemma analysis report is being generated..."}
---------------------------------------------------------

Action Required:
Please check on the patient immediately. If they are unresponsive or experiencing symptoms like chest pain, pressure, shortness of breath, or fainting, call emergency services immediately.

Best regards,
Better-Pulse Wearables Triage Coordinator
"""
        sent_successfully = False
        if recipient_email:
            try:
                msg = MIMEText(body)
                msg["Subject"] = subject
                msg["From"] = settings.SENDER_EMAIL
                msg["To"] = recipient_email

                with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=5) as server:
                    if settings.SMTP_USER and settings.SMTP_PASSWORD:
                        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    server.send_message(msg)
                print(f"[NOTIFICATION] Email alert sent successfully to emergency contact: {recipient_email}")
                sent_successfully = True
            except Exception as e:
                print(f"[NOTIFICATION] SMTP connection failed, falling back to local file logger: {e}")

        # Fallback: write to log file for verification
        try:
            log_dir = settings.STORAGE_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "sent_emails.log"
            
            log_entry = f"""=========================================================
TO: {recipient_email or "No emergency email registered"}
SUBJECT: {subject}
CONTENT:
{body}
=========================================================\n\n"""
            with open(log_file, "a") as f:
                f.write(log_entry)
            print(f"[NOTIFICATION] Alert logged locally to storage/sent_emails.log")
        except Exception as file_err:
            print(f"[NOTIFICATION] Failed to write fallback email log: {file_err}")

        return sent_successfully

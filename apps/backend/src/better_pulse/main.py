import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select

from better_pulse.config import settings
from better_pulse.database import DatabaseConnectionManager, get_session_dependency
from better_pulse.schemas.db_models import Client, FLRound, GlobalModelHistory, SignalSession, AnomalyAlert
from better_pulse.repositories import RoundRepository, ClientRepository, AlertRepository
from better_pulse.server.api import router as api_router
from better_pulse.models.factory import ModelFactory
from better_pulse.models.export_executorch import export_to_executorch

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    db_manager = DatabaseConnectionManager()
    db_manager.init_db()
    
    # Initialize and seed Round 0 if database is empty
    with db_manager.get_session() as session:
        round_repo = RoundRepository(session)
        active_round = round_repo.get_active_round()
        all_rounds = round_repo.get_all_rounds()
        
        if not active_round and not all_rounds:
            print("[STARTUP] No federated rounds found. Seeding Initial Round 0...")
            # 1. Initialize global model safetensors weights
            model_path = ModelFactory.initialize_global_model(round_number=0)
            
            # 2. Export fallback traced .pte model representation
            pte_path = settings.GLOBAL_MODELS_DIR / "global_model_round_0.pte"
            try:
                export_to_executorch(pte_path)
            except Exception as e:
                print(f"[STARTUP] Initial ExecuTorch export failed (using fallback): {e}")
                
            # 3. Save Round 0 metadata
            round_repo.start_new_round(round_number=0, global_model_path=str(model_path))
            print("[STARTUP] Seeding Round 0 complete. Server is ready.")
            
    yield
    print("[SHUTDOWN] Stopping Better-Pulse Server...")

app = FastAPI(
    title="Better-Pulse Coordinator",
    description="Federated Learning & Wearable Cardiac Anomaly Monitoring Framework",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard(session: Session = Depends(get_session_dependency)):
    round_repo = RoundRepository(session)
    client_repo = ClientRepository(session)
    alert_repo = AlertRepository(session)
    
    active_round = round_repo.get_active_round()
    clients = client_repo.get_all()
    history = round_repo.get_global_history()
    active_sessions = alert_repo.get_active_sessions()
    
    round_str = str(active_round.round_number) if active_round else "None (Aggregating or Completed)"
    clients_registered = len(clients)
    active_streams_count = len(active_sessions)
    
    # Extract historical rounds data for chart
    rounds_list = [h.round_number for h in history]
    loss_list = [h.loss for h in history]
    accuracy_list = [h.accuracy for h in history]
    
    # Set default values if list is empty
    latest_acc_str = f"{(history[-1].accuracy * 100):.2f}%" if history and history[-1].accuracy is not None else "N/A"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Better-Pulse Wearable Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {{
                --bg-gradient: linear-gradient(135deg, #090d16 0%, #111827 100%);
                --card-bg: rgba(31, 41, 55, 0.6);
                --text-primary: #f9fafb;
                --text-secondary: #9ca3af;
                --accent-cyan: #06b6d4;
                --accent-indigo: #6366f1;
                --border-color: rgba(255, 255, 255, 0.08);
            }}

            body {{
                font-family: 'Inter', sans-serif;
                background: var(--bg-gradient);
                color: var(--text-primary);
                min-height: 100vh;
                margin: 0;
                padding: 2rem;
                box-sizing: border-box;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}

            header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 2rem;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 1.5rem;
            }}

            h1 {{
                font-weight: 700;
                font-size: 2.25rem;
                margin: 0;
                background: linear-gradient(to right, #06b6d4, #6366f1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}

            .badge {{
                background: rgba(6, 182, 212, 0.2);
                border: 1px solid var(--accent-cyan);
                color: var(--accent-cyan);
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            }}

            .grid-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2.5rem;
            }}

            .card-stat {{
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                display: flex;
                flex-direction: column;
            }}

            .stat-title {{
                color: var(--text-secondary);
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }}

            .stat-value {{
                font-size: 1.875rem;
                font-weight: 700;
                color: var(--text-primary);
            }}

            .grid-main {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 2rem;
                margin-bottom: 2.5rem;
            }}

            .panel {{
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--border-color);
                border-radius: 20px;
                padding: 2rem;
            }}

            .panel h2 {{
                margin-top: 0;
                margin-bottom: 1.5rem;
                font-size: 1.25rem;
                font-weight: 600;
                border-left: 4px solid var(--accent-indigo);
                padding-left: 0.75rem;
            }}

            .chart-container {{
                display: flex;
                gap: 1.5rem;
                margin-top: 1rem;
            }}

            .chart-wrapper {{
                flex: 1;
                min-width: 0;
                position: relative;
                height: 250px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                text-align: left;
            }}

            th, td {{
                padding: 0.75rem 1rem;
                border-bottom: 1px solid var(--border-color);
            }}

            th {{
                color: var(--text-secondary);
                font-weight: 600;
                font-size: 0.875rem;
            }}

            td {{
                font-size: 0.925rem;
            }}

            .client-id {{
                font-family: monospace;
                background: rgba(255, 255, 255, 0.05);
                padding: 0.15rem 0.4rem;
                border-radius: 4px;
                color: #e2e8f0;
            }}

            .timestamp {{
                color: var(--text-secondary);
                font-size: 0.85rem;
            }}

            .grid-alerts {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 2rem;
                margin-top: 2rem;
            }}

            /* Modal & Buttons */
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(8px);
                align-items: center;
                justify-content: center;
            }}
            
            .modal-content {{
                background: #1f2937;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 2rem;
                width: 90%;
                max-width: 600px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                color: var(--text-primary);
                position: relative;
            }}
            
            .close-btn {{
                position: absolute;
                top: 1.5rem;
                right: 1.5rem;
                font-size: 1.5rem;
                color: var(--text-secondary);
                cursor: pointer;
                border: none;
                background: none;
            }}
            
            .close-btn:hover {{
                color: var(--text-primary);
            }}
            
            .btn-report {{
                background: rgba(16, 185, 129, 0.2);
                border: 1px solid #10b981;
                color: #10b981;
                padding: 0.25rem 0.5rem;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 600;
                transition: all 0.2s ease;
            }}
            
            .btn-report:hover {{
                background: #10b981;
                color: #ffffff;
            }}
            
            .badge-alert {
                background: rgba(239, 68, 68, 0.2);
                border: 1px solid #ef4444;
                color: #ef4444;
                padding: 0.15rem 0.4rem;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: bold;
            }
            
            .badge-suppressed {
                background: rgba(156, 163, 175, 0.15);
                border: 1px solid var(--text-secondary);
                color: var(--text-secondary);
                padding: 0.15rem 0.4rem;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: bold;
            }
            
            .badge-activity {
                background: rgba(99, 102, 241, 0.15);
                border: 1px solid var(--accent-indigo);
                color: #a5b4fc;
                padding: 0.15rem 0.4rem;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            
            .report-text {{
                background: rgba(17, 24, 39, 0.8);
                border-radius: 12px;
                padding: 1.5rem;
                font-family: 'Inter', sans-serif;
                font-size: 0.95rem;
                line-height: 1.6;
                white-space: pre-wrap;
                margin-top: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }}
            
            .badge-pulse {{
                display: inline-block;
                width: 10px;
                height: 10px;
                background-color: #ef4444;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 1.5s infinite;
            }}

            @keyframes pulse {{
                0% {{
                    transform: scale(0.9);
                    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
                }}
                70% {{
                    transform: scale(1);
                    box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
                }}
                100% {{
                    transform: scale(0.9);
                    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1>Better-Pulse Wearables Coordinator</h1>
                    <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">Real-Time Patient Anomaly Tracking & Federated Aggregator</p>
                </div>
                <div class="badge">Round {round_str} (OPEN)</div>
            </header>

            <div class="grid-stats">
                <div class="card-stat">
                    <div class="stat-title">Active Wearable Streams</div>
                    <div class="stat-value" id="activeStreamsVal">{active_streams_count}</div>
                </div>
                <div class="card-stat">
                    <div class="stat-title">Registered Clients</div>
                    <div class="stat-value">{clients_registered}</div>
                </div>
                <div class="card-stat">
                    <div class="stat-title">Total Rounds Run</div>
                    <div class="stat-value">{len(history)}</div>
                </div>
                <div class="card-stat">
                    <div class="stat-title">Latest Global Accuracy</div>
                    <div class="stat-value" style="color: var(--accent-cyan);">{latest_acc_str}</div>
                </div>
            </div>

            <div class="grid-main">
                <div class="panel">
                    <h2>Federated Telemetry Progress</h2>
                    <div class="chart-container">
                        <div class="chart-wrapper">
                            <canvas id="lossChart"></canvas>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="accuracyChart"></canvas>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h2>Registered Clients List</h2>
                    <div style="max-height: 250px; overflow-y: auto;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Client ID</th>
                                    <th>Last Active</th>
                                </tr>
                            </thead>
                            <tbody>
                                {"".join(f"<tr><td><span class='client-id'>{c.device_id}</span></td><td><span class='timestamp'>{c.last_active.strftime('%H:%M:%S')}</span></td></tr>" for c in clients) if clients else "<tr><td colspan='2' style='color: var(--text-secondary); text-align:center;'>No registered client devices.</td></tr>"}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="grid-alerts">
                <div class="panel">
                    <h2>Real-Time Wearable Cardiac Anomaly Alerts (Ollama Gemma-4 Assisted)</h2>
                    <div style="max-height: 400px; overflow-y: auto;">
                        <table id="alertsTable">
                            <thead>
                                <tr>
                                    <th>Alert ID</th>
                                    <th>Client Device</th>
                                    <th>Timestamp</th>
                                    <th>Activity State</th>
                                    <th>Est. Heart Rate</th>
                                    <th>Confidence</th>
                                    <th>Clinical Analysis</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td colspan="7" style="text-align:center; color: var(--text-secondary);">Connecting to system metrics...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Gemma Diagnosis Modal -->
        <div id="reportModal" class="modal">
            <div class="modal-content">
                <button class="close-btn" onclick="closeModal()">&times;</button>
                <h2 id="modalTitle" style="border-left: 4px solid #10b981; padding-left: 0.75rem; margin-top: 0;">Gemma-4 Assisted Clinical Report</h2>
                <p id="modalMeta" style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem;"></p>
                <div id="modalBody" class="report-text"></div>
                <div style="text-align: right; margin-top: 1.5rem;">
                    <button class="btn-report" onclick="closeModal()" style="padding: 0.5rem 1rem;">Close Report</button>
                </div>
            </div>
        </div>

        <script>
            // Plot loss history
            const rounds = {rounds_list};
            const lossData = {loss_list};
            const accuracyData = {accuracy_list};

            const ctxLoss = document.getElementById('lossChart').getContext('2d');
            new Chart(ctxLoss, {{
                type: 'line',
                data: {{
                    labels: rounds,
                    datasets: [{{
                        label: 'Global Weighted Loss',
                        data: lossData,
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#9ca3af' }} }},
                        y: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#9ca3af' }} }}
                    }}
                }}
            }});

            const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
            new Chart(ctxAcc, {{
                type: 'line',
                data: {{
                    labels: rounds,
                    datasets: [{{
                        label: 'Global Weighted Accuracy',
                        data: accuracyData,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#9ca3af' }} }},
                        y: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#9ca3af' }}, min: 0.0, max: 1.0 }}
                    }}
                }}
            }});

            // Polling and Modal functions
            async function pollStats() {{
                try {{
                    const statusResp = await fetch('/api/status');
                    const statusData = await statusResp.json();
                    
                    document.getElementById('activeStreamsVal').innerText = statusData.active_streams_count;
                    
                    const alertsResp = await fetch('/api/alerts');
                    const alerts = await alertsResp.json();
                    
                    const tbody = document.querySelector('#alertsTable tbody');
                    if (alerts.length === 0) {{
                        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-secondary);">No cardiac anomalies detected yet. Monitoring live signals...</td></tr>`;
                        return;
                    }}
                    
                    let html = '';
                    for (const a of alerts) {{
                        const reportBtn = a.gemma_report 
                            ? `<button class="btn-report" onclick="showReport(${{a.id}})">View AI Report</button>` 
                            : (a.is_suppressed 
                                ? `<span style="color: var(--text-secondary); font-style: italic;">Suppressed (Exercise)</span>` 
                                : `<span style="color: var(--text-secondary); font-style: italic;"><span class="badge-pulse"></span>Analyzing...</span>`);
                        
                        const timeStr = new Date(a.timestamp).toLocaleTimeString();
                        const badgeClass = a.is_suppressed ? 'badge-suppressed' : 'badge-alert';
                        const badgeText = a.is_suppressed ? 'Suppressed' : `Alert #${{a.id}}`;
                        
                        html += `<tr>
                            <td><span class="${{badgeClass}}">${{badgeText}}</span></td>
                            <td><span class="client-id">${{a.client_id}}</span></td>
                            <td><span class="timestamp">${{timeStr}}</span></td>
                            <td><span class="badge-activity">${{a.activity_state}}</span></td>
                            <td style="color: ${{a.is_suppressed ? 'var(--text-secondary)' : '#ef4444'}}; font-weight: bold;">${{a.heart_rate}} BPM</td>
                            <td style="font-weight: 600; color: ${{a.is_suppressed ? 'var(--text-secondary)' : '#f59e0b'}};">${{(a.confidence * 100).toFixed(1)}}%</td>
                            <td>${{reportBtn}}</td>
                        </tr>`;
                    }}
                    tbody.innerHTML = html;
                    
                    window.gemmaReports = window.gemmaReports || {{}};
                    for (const a of alerts) {{
                        if (a.gemma_report) {{
                            window.gemmaReports[a.id] = {{
                                report: a.gemma_report,
                                client_id: a.client_id,
                                heart_rate: a.heart_rate,
                                confidence: a.confidence,
                                timestamp: a.timestamp
                            }};
                        }}
                    }}
                }} catch (e) {{
                    console.error("Polling error:", e);
                }}
            }}
            
            function showReport(alertId) {{
                const reportData = window.gemmaReports[alertId];
                if (!reportData) return;
                
                document.getElementById('modalTitle').innerText = `Gemma-4 Assisted Clinical Report (Alert #${{alertId}})`;
                document.getElementById('modalMeta').innerHTML = `
                    <strong>Device ID:</strong> <span class="client-id">${{reportData.client_id}}</span> | 
                    <strong>Heart Rate:</strong> ${{reportData.heart_rate}} BPM | 
                    <strong>Confidence:</strong> ${{(reportData.confidence * 100).toFixed(1)}}%
                `;
                document.getElementById('modalBody').innerText = reportData.report;
                document.getElementById('reportModal').style.display = 'flex';
            }}
            
            function closeModal() {{
                document.getElementById('reportModal').style.display = 'none';
            }}
            
            // Poll immediately and set intervals
            pollStats();
            setInterval(pollStats, 3000);
        </script>
    </body>
    </html>
    """
    return html_content

def main():
    print(f"[STARTUP] Launching Better-Pulse Server on {settings.HOST}:{settings.PORT}...")
    uvicorn.run("better_pulse.main:app", host=settings.HOST, port=settings.PORT, reload=True)

if __name__ == "__main__":
    main()

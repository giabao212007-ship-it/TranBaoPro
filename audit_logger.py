import datetime
from typing import List, Dict, Any

class AuditLogger:
    """
    Audit Trail Logger for AI Agent.
    Tracks every action, tool execution, rule application, data transformations,
    and maintains provenance linkage to raw transaction IDs.
    """
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def log(self, step: str, skill: str, action: str, details: Dict[str, Any], status: str = "SUCCESS"):
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "step": step,
            "skill": skill,
            "action": action,
            "status": status,
            "details": details
        }
        self.logs.append(entry)
        return entry

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs

    def export_summary(self) -> str:
        lines = ["=== AI AGENT AUDIT TRAIL LOG ==="]
        for l in self.logs:
            lines.append(f"[{l['timestamp']}] [{l['step']}] [{l['skill']}] {l['action']} -> {l['status']}")
            for k, v in l['details'].items():
                lines.append(f"    - {k}: {v}")
        return "\n".join(lines)


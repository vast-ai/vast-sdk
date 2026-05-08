import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestStatus:
    """Observable status tracker for a serverless request lifecycle."""
    status: str = "New"
    create_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    complete_time: Optional[float] = None
    req_idx: int = 0
    # Worker URL once the autoscaler has allocated one. Updated each
    # time the request lands on a worker, including on retry. Lets
    # observers (e.g. dashboards / progress UIs) show which worker
    # picked up the job before the response itself comes back.
    worker_url: Optional[str] = None

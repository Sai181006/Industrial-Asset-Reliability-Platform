"""Deterministic, configurable RUL risk and workflow decisions."""

from __future__ import annotations

from dataclasses import dataclass
import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Alert, Prediction, WorkOrder


SEVERITY_RANK = {"NORMAL": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@dataclass(frozen=True)
class RiskThresholds:
    critical: float = 10.0
    high: float = 30.0
    medium: float = 70.0

    def __post_init__(self) -> None:
        if not 0 <= self.critical <= self.high <= self.medium:
            raise ValueError("risk thresholds must satisfy 0 <= critical <= high <= medium")

    @classmethod
    def from_environment(cls) -> "RiskThresholds":
        return cls(
            critical=float(os.getenv("RUL_CRITICAL_THRESHOLD", "10")),
            high=float(os.getenv("RUL_HIGH_THRESHOLD", "30")),
            medium=float(os.getenv("RUL_MEDIUM_THRESHOLD", "70")),
        )

    def severity(self, operational_rul: float) -> str:
        if operational_rul <= self.critical:
            return "CRITICAL"
        if operational_rul <= self.high:
            return "HIGH"
        if operational_rul <= self.medium:
            return "MEDIUM"
        return "NORMAL"


@dataclass(frozen=True)
class DecisionResult:
    severity: str
    alert_id: int | None
    work_order_id: int | None


def apply_decision(
    db: Session,
    prediction: Prediction,
    thresholds: RiskThresholds,
) -> DecisionResult:
    severity = thresholds.severity(prediction.operational_rul)
    prediction.risk_severity = severity
    if severity == "NORMAL":
        db.commit()
        return DecisionResult(severity=severity, alert_id=None, work_order_id=None)

    alert = db.scalar(
        select(Alert).where(Alert.asset_id == prediction.asset_id, Alert.active.is_(True))
    )
    if alert is None:
        alert = Alert(
            asset_id=prediction.asset_id,
            prediction_id=prediction.id,
            severity=severity,
            status="OPEN",
            active=True,
        )
        db.add(alert)
        db.flush()
    elif SEVERITY_RANK[severity] > SEVERITY_RANK[alert.severity]:
        alert.severity = severity
        alert.prediction_id = prediction.id

    work_order = db.scalar(select(WorkOrder).where(WorkOrder.alert_id == alert.id))
    if severity == "CRITICAL" and work_order is None:
        work_order = WorkOrder(asset_id=prediction.asset_id, alert_id=alert.id, status="OPEN")
        db.add(work_order)
        db.flush()
    db.commit()
    return DecisionResult(
        severity=severity,
        alert_id=alert.id,
        work_order_id=work_order.id if work_order is not None else None,
    )


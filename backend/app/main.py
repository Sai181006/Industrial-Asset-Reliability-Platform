"""Minimal synchronous inference API; persistence and workflows are deferred."""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import os

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.decision import RiskThresholds, apply_decision
from backend.app.inference import RULInferenceService
from backend.app.models import Alert, Asset, Prediction, TelemetryPointRecord, WorkOrder
from backend.app.schemas import (
    AssetCreate,
    AssetDetailResponse,
    AssetResponse,
    AssetUpdate,
    AlertResponse,
    AlertStatusUpdate,
    DecisionResponse,
    RULPredictionRequest,
    RULPredictionResponse,
    StoredPredictionResponse,
    StoredTelemetryResponse,
    TelemetryIngestResponse,
    TelemetryPoint,
    WorkOrderResponse,
    WorkOrderUpdate,
)
from ml.baseline import FEATURE_COLUMNS


app = FastAPI(
    title="Industrial Asset Reliability Platform",
    version="0.1.0",
    description="Phase 4 synchronous promoted-model inference boundary.",
)

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_inference_service() -> RULInferenceService:
    return RULInferenceService()


@lru_cache(maxsize=1)
def get_risk_thresholds() -> RiskThresholds:
    return RiskThresholds.from_environment()


@app.post("/v1/predictions/rul", response_model=RULPredictionResponse)
def predict_rul_endpoint(
    request: RULPredictionRequest,
    service: RULInferenceService = Depends(get_inference_service),
) -> RULPredictionResponse:
    return service.predict(request)


@app.post("/v1/assets", response_model=AssetResponse, status_code=status.HTTP_201_CREATED)
def create_asset(request: AssetCreate, db: Session = Depends(get_db)) -> AssetResponse:
    if db.get(Asset, request.asset_id) is not None:
        raise HTTPException(status_code=409, detail="asset already exists")
    asset = Asset(id=request.asset_id, name=request.name)
    db.add(asset)
    db.commit()
    return AssetResponse(asset_id=asset.id, name=asset.name)


@app.get("/v1/assets", response_model=list[AssetResponse])
def list_assets(db: Session = Depends(get_db)) -> list[AssetResponse]:
    assets = db.scalars(select(Asset).order_by(Asset.id)).all()
    return [AssetResponse(asset_id=asset.id, name=asset.name) for asset in assets]


def _stored_prediction(prediction: Prediction) -> StoredPredictionResponse:
    return StoredPredictionResponse.model_validate(prediction, from_attributes=True)


@app.get("/v1/assets/{asset_id}", response_model=AssetDetailResponse)
def get_asset_detail(asset_id: str, db: Session = Depends(get_db)) -> AssetDetailResponse:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="asset not found")
    telemetry_count = db.scalar(
        select(func.count()).select_from(TelemetryPointRecord).where(TelemetryPointRecord.asset_id == asset_id)
    )
    prediction_count = db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.asset_id == asset_id)
    )
    latest_telemetry = db.scalar(
        select(TelemetryPointRecord)
        .where(TelemetryPointRecord.asset_id == asset_id)
        .order_by(TelemetryPointRecord.cycle.desc())
        .limit(1)
    )
    latest_prediction = db.scalar(
        select(Prediction)
        .where(Prediction.asset_id == asset_id)
        .order_by(Prediction.window_end_cycle.desc())
        .limit(1)
    )
    active_alert = db.scalar(select(Alert).where(Alert.asset_id == asset_id, Alert.active.is_(True)))
    active_order = None
    if active_alert is not None:
        active_order = db.scalar(select(WorkOrder).where(WorkOrder.alert_id == active_alert.id))
    return AssetDetailResponse(
        asset_id=asset.id,
        name=asset.name,
        telemetry_count=int(telemetry_count or 0),
        prediction_count=int(prediction_count or 0),
        latest_cycle=latest_telemetry.cycle if latest_telemetry else None,
        latest_prediction=_stored_prediction(latest_prediction) if latest_prediction else None,
        active_alert=AlertResponse.model_validate(active_alert, from_attributes=True) if active_alert else None,
        active_work_order=WorkOrderResponse.model_validate(active_order, from_attributes=True) if active_order else None,
    )


@app.patch("/v1/assets/{asset_id}", response_model=AssetResponse)
def update_asset(
    asset_id: str, request: AssetUpdate, db: Session = Depends(get_db)
) -> AssetResponse:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="asset not found")
    asset.name = request.name
    db.commit()
    return AssetResponse(asset_id=asset.id, name=asset.name)


@app.get("/v1/assets/{asset_id}/telemetry", response_model=list[StoredTelemetryResponse])
def recent_telemetry(
    asset_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[StoredTelemetryResponse]:
    if db.get(Asset, asset_id) is None:
        raise HTTPException(status_code=404, detail="asset not found")
    records = list(
        db.scalars(
            select(TelemetryPointRecord)
            .where(TelemetryPointRecord.asset_id == asset_id)
            .order_by(TelemetryPointRecord.cycle.desc())
            .limit(limit)
        )
    )
    records.reverse()
    return [StoredTelemetryResponse.model_validate(record, from_attributes=True) for record in records]


@app.get("/v1/assets/{asset_id}/predictions", response_model=list[StoredPredictionResponse])
def prediction_history(
    asset_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[StoredPredictionResponse]:
    if db.get(Asset, asset_id) is None:
        raise HTTPException(status_code=404, detail="asset not found")
    predictions = db.scalars(
        select(Prediction)
        .where(Prediction.asset_id == asset_id)
        .order_by(Prediction.window_end_cycle.desc())
        .limit(limit)
    ).all()
    return [_stored_prediction(prediction) for prediction in predictions]


def _same_telemetry(record: TelemetryPointRecord, point: TelemetryPoint) -> bool:
    return all(getattr(record, field) == getattr(point, field) for field in ["cycle", *FEATURE_COLUMNS])


@app.post("/v1/assets/{asset_id}/telemetry", response_model=TelemetryIngestResponse)
def ingest_telemetry(
    asset_id: str,
    point: TelemetryPoint,
    db: Session = Depends(get_db),
    service: RULInferenceService = Depends(get_inference_service),
    thresholds: RiskThresholds = Depends(get_risk_thresholds),
) -> TelemetryIngestResponse:
    if db.get(Asset, asset_id) is None:
        raise HTTPException(status_code=404, detail="asset not found")

    existing = db.scalar(
        select(TelemetryPointRecord).where(
            TelemetryPointRecord.asset_id == asset_id,
            TelemetryPointRecord.cycle == point.cycle,
        )
    )
    created = existing is None
    if existing is not None and not _same_telemetry(existing, point):
        raise HTTPException(status_code=409, detail="cycle already exists with different telemetry")
    if existing is None:
        existing = TelemetryPointRecord(asset_id=asset_id, **point.model_dump())
        db.add(existing)
        db.commit()

    latest = list(
        db.scalars(
            select(TelemetryPointRecord)
            .where(TelemetryPointRecord.asset_id == asset_id)
            .order_by(TelemetryPointRecord.cycle.desc())
            .limit(30)
        )
    )
    latest.reverse()
    prediction_response = None
    decision_response = None
    if len(latest) == 30:
        cycles = [record.cycle for record in latest]
        if cycles == list(range(cycles[0], cycles[0] + 30)):
            model_version = service.manifest["version"]
            prediction = db.scalar(
                select(Prediction).where(
                    Prediction.asset_id == asset_id,
                    Prediction.window_end_cycle == cycles[-1],
                    Prediction.model_version == model_version,
                )
            )
            if prediction is None:
                request = RULPredictionRequest(
                    asset_id=asset_id,
                    telemetry=[
                        TelemetryPoint(
                            cycle=record.cycle,
                            **{field: getattr(record, field) for field in FEATURE_COLUMNS},
                        )
                        for record in latest
                    ],
                )
                calculated = service.predict(request)
                prediction = Prediction(
                    asset_id=asset_id,
                    window_start_cycle=calculated.window_start_cycle,
                    window_end_cycle=calculated.window_end_cycle,
                    raw_rul=calculated.raw_rul,
                    operational_rul=calculated.operational_rul,
                    model_id=calculated.model_id,
                    model_version=calculated.model_version,
                )
                db.add(prediction)
                db.commit()
            decision = apply_decision(db, prediction, thresholds)
            prediction_response = RULPredictionResponse(
                asset_id=prediction.asset_id,
                window_start_cycle=prediction.window_start_cycle,
                window_end_cycle=prediction.window_end_cycle,
                raw_rul=prediction.raw_rul,
                operational_rul=prediction.operational_rul,
                model_id=prediction.model_id,
                model_version=prediction.model_version,
            )
            decision_response = DecisionResponse(
                severity=decision.severity,
                alert_id=decision.alert_id,
                work_order_id=decision.work_order_id,
            )

    return TelemetryIngestResponse(
        asset_id=asset_id,
        cycle=point.cycle,
        created=created,
        prediction=prediction_response,
        decision=decision_response,
    )


ALERT_TRANSITIONS = {
    "OPEN": {"ACKNOWLEDGED"},
    "ACKNOWLEDGED": {"RESOLVED"},
    "RESOLVED": set(),
}
WORK_ORDER_TRANSITIONS = {
    "OPEN": {"ASSIGNED"},
    "ASSIGNED": {"IN_PROGRESS"},
    "IN_PROGRESS": {"RESOLVED"},
    "RESOLVED": set(),
}


@app.get("/v1/alerts", response_model=list[AlertResponse])
def list_alerts(
    asset_id: str | None = None,
    alert_status: str | None = Query(default=None, alias="status"),
    active: bool | None = None,
    db: Session = Depends(get_db),
) -> list[AlertResponse]:
    statement = select(Alert)
    if asset_id is not None:
        statement = statement.where(Alert.asset_id == asset_id)
    if alert_status is not None:
        statement = statement.where(Alert.status == alert_status.upper())
    if active is not None:
        statement = statement.where(Alert.active.is_(active))
    alerts = db.scalars(statement.order_by(Alert.id)).all()
    return [AlertResponse.model_validate(alert, from_attributes=True) for alert in alerts]


@app.patch("/v1/alerts/{alert_id}", response_model=AlertResponse)
def update_alert(
    alert_id: int, request: AlertStatusUpdate, db: Session = Depends(get_db)
) -> AlertResponse:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")
    desired = request.status.upper()
    if desired != alert.status and desired not in ALERT_TRANSITIONS.get(alert.status, set()):
        raise HTTPException(status_code=409, detail="invalid alert state transition")
    if desired == "RESOLVED":
        work_order = db.scalar(select(WorkOrder).where(WorkOrder.alert_id == alert.id))
        if work_order is not None and work_order.status != "RESOLVED":
            raise HTTPException(status_code=409, detail="resolve the work order before the alert")
        alert.active = False
        alert.resolved_at = datetime.now(timezone.utc)
    alert.status = desired
    db.commit()
    return AlertResponse.model_validate(alert, from_attributes=True)


@app.get("/v1/alerts/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: int, db: Session = Depends(get_db)) -> AlertResponse:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return AlertResponse.model_validate(alert, from_attributes=True)


@app.get("/v1/work-orders", response_model=list[WorkOrderResponse])
def list_work_orders(
    asset_id: str | None = None,
    work_order_status: str | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
) -> list[WorkOrderResponse]:
    statement = select(WorkOrder)
    if asset_id is not None:
        statement = statement.where(WorkOrder.asset_id == asset_id)
    if work_order_status is not None:
        statement = statement.where(WorkOrder.status == work_order_status.upper())
    orders = db.scalars(statement.order_by(WorkOrder.id)).all()
    return [WorkOrderResponse.model_validate(order, from_attributes=True) for order in orders]


@app.patch("/v1/work-orders/{work_order_id}", response_model=WorkOrderResponse)
def update_work_order(
    work_order_id: int,
    request: WorkOrderUpdate,
    db: Session = Depends(get_db),
) -> WorkOrderResponse:
    order = db.get(WorkOrder, work_order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="work order not found")
    desired = request.status.upper()
    if desired != order.status and desired not in WORK_ORDER_TRANSITIONS.get(order.status, set()):
        raise HTTPException(status_code=409, detail="invalid work-order state transition")
    if desired == "ASSIGNED" and not (request.assigned_to or order.assigned_to):
        raise HTTPException(status_code=422, detail="assigned_to is required for ASSIGNED")
    if request.assigned_to is not None:
        order.assigned_to = request.assigned_to
    if request.notes is not None:
        order.notes = request.notes
    if desired == "RESOLVED":
        order.resolved_at = datetime.now(timezone.utc)
    order.status = desired
    db.commit()
    return WorkOrderResponse.model_validate(order, from_attributes=True)


@app.get("/v1/work-orders/{work_order_id}", response_model=WorkOrderResponse)
def get_work_order(work_order_id: int, db: Session = Depends(get_db)) -> WorkOrderResponse:
    order = db.get(WorkOrder, work_order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="work order not found")
    return WorkOrderResponse.model_validate(order, from_attributes=True)

"""Strict API contracts for one FD001-compatible telemetry window."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TelemetryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cycle: int = Field(ge=1)
    operational_setting_1: float
    operational_setting_2: float
    operational_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class RULPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1, max_length=128)
    telemetry: list[TelemetryPoint] = Field(min_length=30, max_length=30)

    @model_validator(mode="after")
    def validate_cycle_sequence(self) -> "RULPredictionRequest":
        cycles = [point.cycle for point in self.telemetry]
        expected = list(range(cycles[0], cycles[0] + 30))
        if cycles != expected:
            raise ValueError("telemetry cycles must be ascending and consecutive")
        return self


class RULPredictionResponse(BaseModel):
    asset_id: str
    window_start_cycle: int
    window_end_cycle: int
    raw_rul: float
    operational_rul: float
    model_id: str
    model_version: str


class AssetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)


class AssetResponse(BaseModel):
    asset_id: str
    name: str


class AssetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=255)


class TelemetryIngestResponse(BaseModel):
    asset_id: str
    cycle: int
    created: bool
    prediction: RULPredictionResponse | None = None
    decision: "DecisionResponse | None" = None


class DecisionResponse(BaseModel):
    severity: str
    alert_id: int | None = None
    work_order_id: int | None = None


class AlertResponse(BaseModel):
    id: int
    asset_id: str
    prediction_id: int
    severity: str
    status: str
    active: bool


class AlertStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


class WorkOrderResponse(BaseModel):
    id: int
    asset_id: str
    alert_id: int
    status: str
    assigned_to: str | None
    notes: str | None


class WorkOrderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    assigned_to: str | None = Field(default=None, max_length=128)
    notes: str | None = Field(default=None, max_length=2000)


class StoredTelemetryResponse(TelemetryPoint):
    id: int
    asset_id: str


class StoredPredictionResponse(BaseModel):
    id: int
    asset_id: str
    window_start_cycle: int
    window_end_cycle: int
    raw_rul: float
    operational_rul: float
    risk_severity: str
    model_id: str
    model_version: str


class AssetDetailResponse(BaseModel):
    asset_id: str
    name: str
    telemetry_count: int
    prediction_count: int
    latest_cycle: int | None
    latest_prediction: StoredPredictionResponse | None
    active_alert: AlertResponse | None
    active_work_order: WorkOrderResponse | None

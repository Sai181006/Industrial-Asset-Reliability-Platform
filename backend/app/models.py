"""Minimal persisted entities for assets, telemetry, and predictions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    telemetry: Mapped[list["TelemetryPointRecord"]] = relationship(back_populates="asset")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="asset")


class TelemetryPointRecord(Base):
    __tablename__ = "telemetry_points"
    __table_args__ = (UniqueConstraint("asset_id", "cycle", name="uq_telemetry_asset_cycle"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    cycle: Mapped[int] = mapped_column(Integer)
    operational_setting_1: Mapped[float] = mapped_column(Float)
    operational_setting_2: Mapped[float] = mapped_column(Float)
    operational_setting_3: Mapped[float] = mapped_column(Float)
    sensor_1: Mapped[float] = mapped_column(Float)
    sensor_2: Mapped[float] = mapped_column(Float)
    sensor_3: Mapped[float] = mapped_column(Float)
    sensor_4: Mapped[float] = mapped_column(Float)
    sensor_5: Mapped[float] = mapped_column(Float)
    sensor_6: Mapped[float] = mapped_column(Float)
    sensor_7: Mapped[float] = mapped_column(Float)
    sensor_8: Mapped[float] = mapped_column(Float)
    sensor_9: Mapped[float] = mapped_column(Float)
    sensor_10: Mapped[float] = mapped_column(Float)
    sensor_11: Mapped[float] = mapped_column(Float)
    sensor_12: Mapped[float] = mapped_column(Float)
    sensor_13: Mapped[float] = mapped_column(Float)
    sensor_14: Mapped[float] = mapped_column(Float)
    sensor_15: Mapped[float] = mapped_column(Float)
    sensor_16: Mapped[float] = mapped_column(Float)
    sensor_17: Mapped[float] = mapped_column(Float)
    sensor_18: Mapped[float] = mapped_column(Float)
    sensor_19: Mapped[float] = mapped_column(Float)
    sensor_20: Mapped[float] = mapped_column(Float)
    sensor_21: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    asset: Mapped[Asset] = relationship(back_populates="telemetry")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "asset_id", "window_end_cycle", "model_version", name="uq_prediction_asset_cycle_model"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    window_start_cycle: Mapped[int] = mapped_column(Integer)
    window_end_cycle: Mapped[int] = mapped_column(Integer)
    raw_rul: Mapped[float] = mapped_column(Float)
    operational_rul: Mapped[float] = mapped_column(Float)
    model_id: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    risk_severity: Mapped[str] = mapped_column(String(16), default="NORMAL")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    asset: Mapped[Asset] = relationship(back_populates="predictions")


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        Index(
            "uq_alert_one_active_per_asset",
            "asset_id",
            unique=True,
            postgresql_where=text("active = true"),
            sqlite_where=text("active = 1"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False)
    severity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="OPEN")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkOrder(Base):
    __tablename__ = "work_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id"), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    assigned_to: Mapped[str | None] = mapped_column(String(128), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

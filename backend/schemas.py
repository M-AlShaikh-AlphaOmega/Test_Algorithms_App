"""
Pydantic schemas for request/response validation.

These models define the contract for API inputs and outputs,
ensuring type safety and automatic documentation.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


class PatientState(str, Enum):
    """Possible patient medication states."""
    ON = "ON"
    OFF = "OFF"
    UNKNOWN = "UNKNOWN"


class SensorDataStats(BaseModel):
    """Statistical summary of the sensor data."""
    
    sample_count: int = Field(
        ..., 
        description="Number of samples in the uploaded data"
    )
    duration_seconds: float = Field(
        ..., 
        description="Estimated duration of recording in seconds"
    )
    mean_magnitude: float = Field(
        ..., 
        description="Mean acceleration magnitude (m/s²)"
    )
    std_magnitude: float = Field(
        ..., 
        description="Standard deviation of acceleration magnitude"
    )
    movement_intensity: float = Field(
        ..., 
        description="Normalized movement intensity score (0-1)"
    )


class DetectionResponse(BaseModel):
    """Response model for state detection endpoint."""
    
    filename: str = Field(
        ..., 
        description="Name of the uploaded CSV file"
    )
    detected_state: PatientState = Field(
        ..., 
        description="Detected patient state: ON, OFF, or UNKNOWN"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score of the detection (0.0 to 1.0)"
    )
    explanation: str = Field(
        ..., 
        description="Human-readable explanation of the detection result"
    )
    sensor_stats: SensorDataStats = Field(
        ..., 
        description="Statistical summary of the analyzed sensor data"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "patient_001_session.csv",
                "detected_state": "ON",
                "confidence": 0.82,
                "explanation": "Movement patterns indicate controlled motor function consistent with ON state. Low tremor activity and smooth movement transitions detected.",
                "sensor_stats": {
                    "sample_count": 250,
                    "duration_seconds": 10.0,
                    "mean_magnitude": 9.85,
                    "std_magnitude": 0.45,
                    "movement_intensity": 0.35
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., 
        description="Current health status of the API"
    )
    version: str = Field(
        ..., 
        description="API version"
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether the detection model is loaded and ready"
    )
    message: str = Field(
        ..., 
        description="Additional status information"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(
        ..., 
        description="Error type or code"
    )
    detail: str = Field(
        ..., 
        description="Detailed error message"
    )
    suggestion: Optional[str] = Field(
        None, 
        description="Suggested action to resolve the error"
    )

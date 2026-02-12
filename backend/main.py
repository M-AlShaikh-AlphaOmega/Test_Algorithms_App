"""
aCare Parkinson's State Detection API

A FastAPI backend for detecting patient medication states (ON/OFF)
from wearable sensor data. Part of the aCare project for Parkinson's
disease monitoring.

Run with: uvicorn main:app --reload
Swagger UI: http://localhost:8080/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from config import settings
from schemas import (
    DetectionResponse, 
    HealthCheckResponse, 
    ErrorResponse,
    SensorDataStats,
    PatientState
)
from utils import (
    parse_csv_content, 
    validate_sensor_data, 
    compute_sensor_statistics,
    DataValidationError
)
from detector import detect_patient_state, get_detector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Initialize resources on startup, cleanup on shutdown."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    detector = get_detector()
    logger.info(f"Detector initialized: ready={detector.is_ready}")
    yield
    logger.info("Shutting down API")


# Initialize FastAPI application
app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    description="""
## aCare Parkinson's State Detection API

This API analyzes accelerometer data from wearable devices to detect 
the current medication state (ON/OFF) of Parkinson's disease patients.

### Features
- **CSV Upload**: Upload sensor data files for analysis
- **State Detection**: Detect ON, OFF, or UNKNOWN states
- **Detailed Statistics**: Get sensor data analysis metrics

### Endpoints
- `POST /detect` - Upload CSV and get state detection
- `GET /health` - Check API health status

### Data Format
Upload CSV files with accelerometer data containing columns:
- `acc_x` (or `x`, `accel_x`, `accelerometer_x`)
- `acc_y` (or `y`, `accel_y`, `accelerometer_y`)
- `acc_z` (or `z`, `accel_z`, `accelerometer_z`)

Data should be 5-10 seconds duration at ~25Hz sampling rate.

### Note
This is a development version using placeholder detection logic.
Production deployments will use validated ML models.
    """,
    version=settings.app_version,
    contact={
        "name": "aCare Development Team",
        "email": "dev@acare.health"
    },
    license_info={
        "name": "Proprietary",
    },
    docs_url="/docs",
    redoc_url="/redoc"
)


# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(DataValidationError)
async def validation_exception_handler(request, exc: DataValidationError):
    """Handle data validation errors with helpful messages."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="data_validation_error",
            detail=str(exc),
            suggestion="Ensure your CSV contains accelerometer columns (acc_x, acc_y, acc_z) with 5-10 seconds of numeric data at ~25Hz."
        ).model_dump()
    )


# Endpoints
@app.get(
    "/",
    summary="API Root",
    description="Welcome endpoint with API information",
    tags=["General"]
)
async def root():
    """Return API welcome message and basic info."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check API health status and model readiness",
    tags=["General"]
)
async def health_check():
    """
    Check the health status of the API.
    
    Returns information about:
    - Overall API status
    - Whether the prediction model is loaded
    - API version
    """
    detector = get_detector()
    model_ready = detector.is_ready
    
    if model_ready:
        return HealthCheckResponse(
            status="healthy",
            version=settings.app_version,
            model_loaded=True,
            message="API is operational. Detector is ready for inference."
        )
    else:
        return HealthCheckResponse(
            status="degraded",
            version=settings.app_version,
            model_loaded=False,
            message="API is running but detection model is not loaded."
        )


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        200: {
            "description": "Successful detection",
            "model": DetectionResponse
        },
        400: {
            "description": "Invalid file format",
            "model": ErrorResponse
        },
        422: {
            "description": "Data validation error",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    },
    summary="Detect Patient State",
    description="""
Upload a CSV file containing accelerometer sensor data to detect
the patient's current medication state.

### Input Requirements
- **File Format**: CSV
- **Required Columns**: acc_x, acc_y, acc_z (or variants)
- **Data Duration**: 5-10 seconds
- **Sampling Rate**: ~25Hz recommended

### Output
Returns the detected state (ON/OFF/UNKNOWN) with confidence score
and detailed explanation of the analysis.
    """,
    tags=["Detection"]
)
async def detect_state(
    file: UploadFile = File(
        ...,
        description="CSV file containing accelerometer sensor data"
    )
):
    """
    Analyze uploaded sensor data and detect patient state.

    The detection pipeline:
    1. Parse CSV file
    2. Validate sensor data columns and quality
    3. Compute statistical features
    4. Run detection model
    5. Return state with confidence and explanation
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error="invalid_file_type",
                detail=f"Expected CSV file, got: {file.filename}",
                suggestion="Please upload a file with .csv extension"
            ).model_dump()
        )
    
    logger.info(f"Processing file: {file.filename}")
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="empty_file",
                    detail="The uploaded file is empty",
                    suggestion="Upload a CSV file with sensor data"
                ).model_dump()
            )
        
        # Parse CSV
        df = parse_csv_content(content, file.filename)
        logger.info(f"Parsed CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        # Validate and prepare data (returns acc + optional gyro columns)
        validated_df, ax_col, ay_col, az_col, gx_col, gy_col, gz_col = validate_sensor_data(df, file.filename)
        has_gyro = gx_col is not None
        col_info = f"[{ax_col}, {ay_col}, {az_col}]"
        if has_gyro:
            col_info += f" + [{gx_col}, {gy_col}, {gz_col}]"
        logger.info(f"Validated data: {len(validated_df)} samples using columns {col_info}")

        # Compute statistics (uses acc columns for summary stats)
        sensor_stats = compute_sensor_statistics(validated_df, ax_col, ay_col, az_col)
        logger.info(f"Computed statistics: movement_intensity={sensor_stats['movement_intensity']:.3f}")

        # Attach raw arrays for ML feature extraction
        sensor_stats['_raw_ax'] = validated_df[ax_col].values
        sensor_stats['_raw_ay'] = validated_df[ay_col].values
        sensor_stats['_raw_az'] = validated_df[az_col].values
        if has_gyro:
            sensor_stats['_raw_gx'] = validated_df[gx_col].values
            sensor_stats['_raw_gy'] = validated_df[gy_col].values
            sensor_stats['_raw_gz'] = validated_df[gz_col].values

        # Run prediction
        detection = detect_patient_state(sensor_stats)
        logger.info(f"Detection: {detection.state.value} (confidence: {detection.confidence:.2f})")
        
        # Build response
        return DetectionResponse(
            filename=file.filename,
            detected_state=detection.state,
            confidence=detection.confidence,
            explanation=detection.explanation,
            sensor_stats=SensorDataStats(
                sample_count=sensor_stats['sample_count'],
                duration_seconds=round(sensor_stats['duration_seconds'], 2),
                mean_magnitude=round(sensor_stats['mean_magnitude'], 3),
                std_magnitude=round(sensor_stats['std_magnitude'], 3),
                movement_intensity=round(sensor_stats['movement_intensity'], 3)
            )
        )
        
    except DataValidationError:
        # Re-raise to be handled by exception handler
        raise
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Unexpected error processing {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="processing_error",
                detail=f"An unexpected error occurred: {str(e)}",
                suggestion="Please check your file format and try again. If the issue persists, contact support."
            ).model_dump()
        )


@app.get(
    "/config",
    summary="Get Configuration",
    description="Returns current API configuration (non-sensitive)",
    tags=["General"]
)
async def get_config():
    """Return non-sensitive configuration parameters."""
    return {
        "expected_sample_rate_hz": settings.expected_sample_rate_hz,
        "min_samples": settings.min_samples,
        "max_samples": settings.max_samples,
        "min_duration_seconds": settings.min_samples / settings.expected_sample_rate_hz,
        "max_duration_seconds": settings.max_samples / settings.expected_sample_rate_hz,
        "accepted_column_names": {
            "x_axis": settings.acc_x_columns,
            "y_axis": settings.acc_y_columns,
            "z_axis": settings.acc_z_columns
        },
        "confidence_threshold": settings.confidence_threshold
    }


# Main entry point for direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.debug
    )

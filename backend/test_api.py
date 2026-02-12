"""
Tests for the aCare Parkinson's State Detection API.

Uses FastAPI TestClient for synchronous testing without running a server.

Run with: python test_api.py
"""

from fastapi.testclient import TestClient
import json
import sys

# Add backend to path
sys.path.insert(0, '.')

from main import app

# Create test client
client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    print("=" * 50)
    print("TEST: Health Check")
    print("=" * 50)
    
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    print(json.dumps(data, indent=2))
    
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    print("✓ PASSED\n")


def test_config_endpoint():
    """Test the configuration endpoint."""
    print("=" * 50)
    print("TEST: Configuration")
    print("=" * 50)
    
    response = client.get("/config")
    assert response.status_code == 200
    
    data = response.json()
    print(json.dumps(data, indent=2))
    
    assert "expected_sample_rate_hz" in data
    print("✓ PASSED\n")


def test_detect_on_state():
    """Test detection with ON state data."""
    print("=" * 50)
    print("TEST: Detect ON State")
    print("=" * 50)
    
    with open("test_data/test_on_state.csv", "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("test_on_state.csv", f, "text/csv")}
        )
    
    assert response.status_code == 200
    
    data = response.json()
    print(json.dumps(data, indent=2))
    
    assert data["filename"] == "test_on_state.csv"
    assert data["detected_state"] in ["ON", "OFF", "UNKNOWN"]
    assert 0 <= data["confidence"] <= 1
    assert "sensor_stats" in data
    print(f"\n→ Detected: {data['detected_state']} (confidence: {data['confidence']:.2f})")
    print("✓ PASSED\n")


def test_detect_off_state():
    """Test detection with OFF state data."""
    print("=" * 50)
    print("TEST: Detect OFF State")
    print("=" * 50)
    
    with open("test_data/test_off_state.csv", "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("test_off_state.csv", f, "text/csv")}
        )
    
    assert response.status_code == 200
    
    data = response.json()
    print(json.dumps(data, indent=2))
    
    assert data["filename"] == "test_off_state.csv"
    assert data["detected_state"] in ["ON", "OFF", "UNKNOWN"]
    print(f"\n→ Detected: {data['detected_state']} (confidence: {data['confidence']:.2f})")
    print("✓ PASSED\n")


def test_detect_unknown_state():
    """Test detection with ambiguous data."""
    print("=" * 50)
    print("TEST: Detect Unknown State")
    print("=" * 50)
    
    with open("test_data/test_unknown.csv", "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("test_unknown.csv", f, "text/csv")}
        )
    
    assert response.status_code == 200
    
    data = response.json()
    print(json.dumps(data, indent=2))
    
    print(f"\n→ Detected: {data['detected_state']} (confidence: {data['confidence']:.2f})")
    print("✓ PASSED\n")


def test_detect_minimal_columns():
    """Test detection with minimal column names (x, y, z)."""
    print("=" * 50)
    print("TEST: Minimal Column Names")
    print("=" * 50)
    
    with open("test_data/test_minimal_columns.csv", "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("test_minimal_columns.csv", f, "text/csv")}
        )
    
    assert response.status_code == 200
    
    data = response.json()
    print(json.dumps(data, indent=2))
    
    print(f"\n→ Detected: {data['detected_state']} (confidence: {data['confidence']:.2f})")
    print("✓ PASSED\n")


def test_invalid_file_type():
    """Test error handling for non-CSV file."""
    print("=" * 50)
    print("TEST: Invalid File Type")
    print("=" * 50)
    
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    
    assert response.status_code == 400
    
    data = response.json()
    print(json.dumps(data, indent=2))
    
    assert "error" in data["detail"]
    print("✓ PASSED (correctly rejected invalid file)\n")


def test_empty_file():
    """Test error handling for empty file."""
    print("=" * 50)
    print("TEST: Empty File")
    print("=" * 50)
    
    response = client.post(
        "/detect",
        files={"file": ("empty.csv", b"", "text/csv")}
    )
    
    assert response.status_code == 400
    
    data = response.json()
    print(json.dumps(data, indent=2))
    
    print("✓ PASSED (correctly rejected empty file)\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("aCare Parkinson's State Detection API - Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_health_check,
        test_config_endpoint,
        test_detect_on_state,
        test_detect_off_state,
        test_detect_unknown_state,
        test_detect_minimal_columns,
        test_invalid_file_type,
        test_empty_file,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

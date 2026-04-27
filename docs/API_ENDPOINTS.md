# API Endpoints

The project includes a FastAPI server (`apps/api_server/main.py`) that provides the following API endpoints to interact with the surveillance system's data.

## Defined Endpoints

### 1. Root Endpoint
- **Method:** `GET`
- **Path:** `/`
- **Description:** Returns a welcome message and a basic directory structure of the available endpoints.

### 2. Health Check
- **Method:** `GET`
- **Path:** `/health`
- **Description:** Returns `{"status": "ok"}` to verify that the API server is up and running.

### 3. Latest Events
- **Method:** `GET`
- **Path:** `/events/latest`
- **Description:** Returns the 20 most recent surveillance events from the log file (`logs/events.jsonl`).
- **Response Type:** Array of `SurveillanceEvent` objects.

### 4. All Events
- **Method:** `GET`
- **Path:** `/events`
- **Description:** Returns a larger list of surveillance events. 
- **Query Parameters:**
  - `limit` (int, optional): The maximum number of events to return. Defaults to 200. Example: `/events?limit=500`
- **Response Type:** Array of `SurveillanceEvent` objects.

## Built-in Documentation Endpoints

Because the server is built using FastAPI, it automatically generates and serves interactive API documentation at the following paths:

- **Swagger UI:** `GET /docs` (Interactive documentation where you can test the endpoints directly from the browser)
- **ReDoc:** `GET /redoc` (Alternative, clean API documentation format)

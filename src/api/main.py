from fastapi import FastAPI

app = FastAPI(title="Flight Delays API")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "ok", "service": "flight-delays-api"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}

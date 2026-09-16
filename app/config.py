from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    groq_api_key: str
    redis_url: str = ""  # Optional — empty string means caching is disabled

    class Config:
        env_file = ".env"


settings = Settings()

from pydantic_settings import BaseSettings


class Configs(BaseSettings):
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    SAVE_RESULTS: bool = False


configs = Configs()

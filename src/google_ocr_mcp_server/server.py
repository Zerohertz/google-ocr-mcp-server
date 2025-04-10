import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from google.cloud import vision
from google.protobuf.json_format import MessageToDict
from loguru import logger
from mcp.server.fastmcp import FastMCP

from .settings import configs


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, any]]:
    try:
        logger.info("Google OCR MCP server starting up")
        yield {}
    finally:
        logger.info("Google OCR MCP server shut down")


mcp = FastMCP(
    name="google-ocr-mcp-server",
    instructions="Google OCR on images",
    lifespan=lifespan,
    lazy_load=True,
)

_client: vision.ImageAnnotatorClient | None = None


def get_client() -> vision.ImageAnnotatorClient:
    global _client
    if _client is None:
        logger.info("Initializing `ImageAnnotatorClient` ...")
        _client = vision.ImageAnnotatorClient()
        logger.info("`ImageAnnotatorClient` successfully initialized.")
    return _client


def _without_ext(path: str) -> str:
    return ".".join(path.split(".")[:-1])


@mcp.tool()
async def ocr(path: str) -> str:
    """
    Perform Optical Character Recognition (OCR) on the provided image file.

    Args:
        path (str): The absolute file path to the image on which OCR will be performed.

    Returns:
        str: The extracted text from the image.

    Raises:
        Exception: If an error occurs during the OCR process, it will be logged.

    Notes:
        - The function uses Google Cloud Vision API for text detection.
        - If SAVE_RESULTS is enabled, the OCR results will be saved as a JSON file
          in the same directory as the input image, with the same name but a .json extension.
    """
    client = get_client()
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        logger.error(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    if configs.SAVE_RESULTS:
        response_dict = MessageToDict(response._pb)
        _path = _without_ext(path)
        with open(_path + ".json", "w") as file:
            json.dump(response_dict, file, ensure_ascii=False)
    return response.full_text_annotation.text

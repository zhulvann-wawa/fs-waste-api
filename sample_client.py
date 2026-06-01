from pathlib import Path
import sys

import requests


API_URL = "http://localhost:8000/predict"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python sample_client.py path/to/image.jpg")

    image_path = Path(sys.argv[1])
    with image_path.open("rb") as image_file:
        response = requests.post(
            API_URL,
            files={"file": (image_path.name, image_file, "image/jpeg")},
            timeout=30,
        )

    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()

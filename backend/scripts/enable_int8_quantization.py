import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


def main():
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        print("❌ Missing QDRANT_URL or QDRANT_API_KEY in .env")
        return

    collection_name = "arxiv_text"

    print(f"Connecting to Qdrant: {qdrant_url}")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=300,
    )

    print(f"Enabling INT8 scalar quantization on '{collection_name}'...")

    try:
        client.update_collection(
            collection_name=collection_name,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=False,
                )
            )
        )

        print("✅ INT8 scalar quantization enabled.")
        print("Qdrant will now quantize existing vectors in the background.")
        print("Future uploaded vectors will also be quantized automatically.")

    except Exception as e:
        print(f"❌ Failed to enable quantization: {e}")


if __name__ == "__main__":
    main()
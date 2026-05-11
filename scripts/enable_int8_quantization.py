import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

def main():
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL or QDRANT_API_KEY not found in .env")
        return

    print(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    collection_name = "arxiv_text"

    print(f"Enabling Int8 Scalar Quantization on collection '{collection_name}'...")
    
    try:
        # Update the collection to enable Int8 Quantization
        client.update_collection(
            collection_name=collection_name,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True  # Keep the tiny int8 vectors in RAM for ultra-fast speed
                )
            )
        )
        print("✅ Successfully enabled Int8 Quantization!")
        print("Qdrant is now running a background job to compress all your existing vectors.")
        print("You can verify this in the Qdrant Cloud Dashboard -> Collections -> arxiv_text -> Configuration.")
    except Exception as e:
        print(f"❌ Failed to enable quantization: {e}")

if __name__ == "__main__":
    main()

import comfy_endpoint
import asyncio
import base64

async def main():
    result = await comfy_endpoint.generate_image(
        "A scenic photo from a balcony in Tuscany."
    )
    
    # Print entire result so you can inspect structure
    print("Full result:", result)

    # Extract the base64 image string
    image_b64 = result.get("image_base64")
    
    if not image_b64:
        print("No image returned!")
        return
    
    print("\n=== Base64 Image ===\n")
    print(image_b64)

    # OPTIONAL: Save image locally for visual confirmation
    output_path = "generated_image.png"
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    print(f"\nSaved image as: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())

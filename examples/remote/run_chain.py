import qwendpoint
import comfy_endpoint
import asyncio
import base64


def download_base64(data_b64: str, output_path: str = "generated_image.png"):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(data_b64))


async def main():
    story = "<The beginning of the story>\n"
    sentence_count = 0
    while sentence_count < 10:
        llm_result = await qwendpoint.llm_chat(f"You are generating a story sentence by sentence. Please return the sentence that should follow. Only return a SINGLE SENTENCE. DO NOT RETURN MORE THAN ONE SENTENCE. Here is the story so far:\n{story}")
        next_sentence = llm_result["result"]["choices"][0]["message"]["content"]
        story += f"{next_sentence}\n"
        result = await comfy_endpoint.generate_image(
            f"{next_sentence}"
        )
        image_b64 = result.get("result")
        download_base64(image_b64, f"sentence_{sentence_count}.png")
        print(f"Sentence {sentence_count}: {next_sentence}")
        sentence_count += 1

if __name__ == "__main__":
    asyncio.run(main())
import json

input_filename = r"D:\vivek ai\llm_fientuning\sample3.jsonl"
output_filename = r"D:\vivek ai\llm_fientuning\finetuning_data3.jsonl"

# Read JSONL (one JSON object per line)
data = []
with open(input_filename, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# Write as JSONL for Gemini finetuning
with open(output_filename, "w", encoding="utf-8") as outfile:
    for entry in data:
        prompt_text = entry.get("input", "")
        response_text = json.dumps(entry.get("output", {}), ensure_ascii=False)
        new_entry = {
            "prompt": prompt_text,
            "response": response_text
        }
        outfile.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

print(f"Transformation complete! Data saved to {output_filename}")
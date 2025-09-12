import json

input_filename = r"D:\vivek ai\llm_fientuning\receipt3.jsonl"
output_filename = r"d:\vivek ai\llm_fientuning\finetuning_data_gemini_test.jsonl"

with open(input_filename, "r", encoding="utf-8") as infile:
    data = json.load(infile)

with open(output_filename, "w", encoding="utf-8") as outfile:
    for entry in data:
        prompt = entry.get("input", "")
        # If output is missing or empty, use an empty dict
        output_obj = entry.get("output", {})
        # If output is not a dict, make it a string
        if not isinstance(output_obj, dict):
            output_obj = {}
        response = json.dumps(output_obj, ensure_ascii=False)
        gemini_entry = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "model", "parts": [{"text": response}]}
            ]
        }
        outfile.write(json.dumps(gemini_entry, ensure_ascii=False) + "\n")

print(f"Gemini-compatible file saved to {output_filename}")
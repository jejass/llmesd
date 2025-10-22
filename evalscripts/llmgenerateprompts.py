import os
import json
import argparse
from openai import OpenAI


def generate_concept_prompts(
    output_file: str,
    add_target_instruction: bool,
    add_output_constraint: bool,
    add_role_spec: bool,
    add_fewshot: bool,
    add_cot: bool,
    add_context_info: bool,
    num_prompts: int,
    concept: str
):
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("请先设置环境变量 OPENAI_API_KEY")
    """生成遗忘概念的 prompt 列表并保存结果"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # prompt 文件路径
    base_dir = "prompts"
    files = {
        "target_instruction": "targetinstruction.txt",
        "output_constraint": "outputconstraint.txt",
        "role_spec": "rolespec.txt",
        "fewshot": "fewshot.txt",
        "cot": "cot.txt",
        "context_info": "contextinfo.txt",
    }

    # 读取 prompt 元素内容
    contents = {}
    for key, filename in files.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                contents[key] = f.read().strip()
        else:
            contents[key] = ""

    # 拼接最终 prompt
    prompt_parts = []

    if add_role_spec and contents["role_spec"]:
        prompt_parts.append(contents["role_spec"])
    if add_target_instruction and contents["target_instruction"]:
        prompt_parts.append(
            contents["target_instruction"]
            .replace("[target concept]", concept)
            .replace("[number]", str(num_prompts))
        )
    if add_output_constraint and contents["output_constraint"]:
        prompt_parts.append(contents["output_constraint"])
    if add_fewshot and contents["fewshot"]:
        prompt_parts.append(contents["fewshot"])
    if add_cot and contents["cot"]:
        prompt_parts.append(contents["cot"].replace("[number]", str(num_prompts)))
    if add_context_info and contents["context_info"]:
        prompt_parts.append(contents["context_info"])

    final_prompt = "\n\n".join(prompt_parts)

    # 调用 GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert prompt engineer for diffusion models."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.7
    )

    result_text = response.choices[0].message.content.strip()

    # 验证返回是否符合 {"result": [...]} 格式
    try:
        result_json = json.loads(result_text)
        if not isinstance(result_json, dict) or "result" not in result_json or not isinstance(result_json["result"], list):
            raise ValueError("Response JSON does not match expected schema.")
    except Exception as e:
        print("❌ Format validation failed!")
        print("------ Prompt Sent ------")
        print(final_prompt)
        print("------ Response Received ------")
        print(result_text)
        raise e

    # 写入输出文件
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "concept": concept,
            "num_prompts": num_prompts,
            "prompt": final_prompt,
            "response": result_json
        }, f, ensure_ascii=False, indent=4)

    print(f"✅ Successfully generated {len(result_json['result'])} prompts for concept '{concept}'.")
    print(result_json)
    print(f"Result saved to: {output_file}")
    return result_json


def main():
    parser = argparse.ArgumentParser(description="Generate concept-specific forget prompts using GPT-4o")

    parser.add_argument("--output", type=str, required=True, help="Output file path (e.g., outputs/nudity_prompts.json)")
    parser.add_argument("--concept", type=str, required=True, help="Target concept (e.g., 'nudity', 'Van Gogh')")
    parser.add_argument("--num", type=int, default=10, help="Number of prompts to generate")

    parser.add_argument("--target_instruction", action="store_true", help="Include target instruction section")
    parser.add_argument("--output_constraint", action="store_true", help="Include output constraint section")
    parser.add_argument("--role_spec", action="store_true", help="Include role specification section")
    parser.add_argument("--fewshot", action="store_true", help="Include few-shot examples section")
    parser.add_argument("--cot", action="store_true", help="Include chain-of-thought section")
    parser.add_argument("--context_info", action="store_true", help="Include context information section")

    args = parser.parse_args()

    generate_concept_prompts(
        output_file=args.output,
        add_target_instruction=args.target_instruction,
        add_output_constraint=args.output_constraint,
        add_role_spec=args.role_spec,
        add_fewshot=args.fewshot,
        add_cot=args.cot,
        add_context_info=args.context_info,
        num_prompts=args.num,
        concept=args.concept
    )


if __name__ == "__main__":
    main()
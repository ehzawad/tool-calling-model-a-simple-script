"""
NanoAgent-135M Tool Calling Demo
This script demonstrates tool calling capabilities using the NanoAgent-135M model.
"""

import json
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Tool definitions
TOOLS = [
    {
        "name": "get_joke",
        "description": "Fetches a random programming joke from a public API",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "calculate",
        "description": "Performs mathematical calculations on two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The mathematical operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "num1": {
                    "type": "number",
                    "description": "The first number"
                },
                "num2": {
                    "type": "number",
                    "description": "The second number"
                }
            },
            "required": ["operation", "num1", "num2"]
        }
    }
]

TOOL_TEMPLATE = """You are a helpful AI assistant. You have a set of possible functions/tools inside <tools></tools> tags.
Based on question, you may need to make one or more function/tool calls to answer user.

You have access to the following tools/functions:
<tools>{tools}</tools>

For each function call, return a JSON list object with function name and arguments within <tool_call></tool_call> tags.

User question: {question}

Answer:"""


def get_joke():
    """Fetches a random programming joke from the official joke API"""
    try:
        response = requests.get("https://official-joke-api.appspot.com/jokes/programming/random", timeout=5)
        if response.status_code == 200:
            joke_data = response.json()[0]
            return {
                "success": True,
                "joke": f"{joke_data['setup']} - {joke_data['punchline']}"
            }
        return {"success": False, "error": "Failed to fetch joke"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate(operation, num1, num2):
    """Performs mathematical calculations"""
    try:
        num1 = float(num1)
        num2 = float(num2)

        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return {"success": False, "error": "Division by zero"}
            result = num1 / num2
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_tool(tool_name, arguments):
    """Execute a tool based on the tool name and arguments"""
    if tool_name == "get_joke":
        return get_joke()
    elif tool_name == "calculate":
        return calculate(**arguments)
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}


def parse_tool_calls(response_text):
    """Parse tool calls from the model response"""
    tool_calls = []

    # Look for tool_call tags
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, response_text, re.DOTALL)

    for match in matches:
        try:
            # Clean up the match
            match = match.strip()
            # Try to parse as JSON
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            print(f"Failed to parse tool call: {match}")

    return tool_calls


def main():
    print("=" * 80)
    print("NanoAgent-135M Tool Calling Demo")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model_name = "quwsarohi/NanoAgent-135M"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test cases
    test_questions = [
        "Tell me a programming joke",
        "Calculate 25 multiplied by 4"
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        # Format the prompt with tools
        tools_json = json.dumps(TOOLS, indent=2)
        prompt = TOOL_TEMPLATE.format(tools=tools_json, question=question)

        print("\n[Calling NanoAgent model...]")

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part (after the prompt)
        answer_start = full_response.find("Answer:")
        if answer_start != -1:
            model_response = full_response[answer_start + 7:].strip()
        else:
            model_response = full_response[len(prompt):].strip()

        print(f"\n[Model Response]:\n{model_response}")

        # Parse and execute tool calls
        tool_calls = parse_tool_calls(model_response)

        if tool_calls:
            print(f"\n[Detected {len(tool_calls)} tool call(s)]")

            for i, tool_call in enumerate(tool_calls, 1):
                if isinstance(tool_call, list):
                    # If it's a list of tool calls
                    for j, tc in enumerate(tool_call, 1):
                        tool_name = tc.get("name")
                        arguments = tc.get("arguments", {})
                        print(f"\n[Tool Call {j}]:")
                        print(f"  Tool: {tool_name}")
                        print(f"  Arguments: {json.dumps(arguments)}")

                        # Execute the tool
                        result = execute_tool(tool_name, arguments)
                        print(f"\n[Tool Result {j}]:")
                        print(f"  {json.dumps(result, indent=2)}")
                else:
                    # Single tool call
                    tool_name = tool_call.get("name")
                    arguments = tool_call.get("arguments", {})
                    print(f"\n[Tool Call {i}]:")
                    print(f"  Tool: {tool_name}")
                    print(f"  Arguments: {json.dumps(arguments)}")

                    # Execute the tool
                    result = execute_tool(tool_name, arguments)
                    print(f"\n[Tool Result {i}]:")
                    print(f"  {json.dumps(result, indent=2)}")
        else:
            print("\n[No tool calls detected in response]")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
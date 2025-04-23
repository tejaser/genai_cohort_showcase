import json
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


# Enum for type safety and clarity
class DifficultyLevel(str, Enum):
	ONE = "1"
	TWO = "2"
	THREE = "3"
	FOUR = "4"
	FIVE = "5"


class TaskType(str, Enum):
	WRITING = "Writing"
	QUESTIONS = "Questions"
	MATH = "Math"
	ROLEPLAY = "Roleplay"
	ANALYSIS = "Analysis"
	CREATIVITY = "Creativity"
	CODING = "Coding"
	EDUCATION = "Education"
	RESEARCH = "Research"
	TRANSLATION = "Translation"


# Descriptive metadata
DIFFICULTY_DETAILS = {
	DifficultyLevel.ONE: "This query is just a greeting or basic interaction.",
	DifficultyLevel.TWO: "Simple question, answerable in one sentence.",
	DifficultyLevel.THREE: "Needs basic logic, short code, or brief reasoning.",
	DifficultyLevel.FOUR: "Requires paragraph answers, more logic, or detailed coding.",
	DifficultyLevel.FIVE: "Complex reasoning, intricate logic, long answers or deep coding.",
}

TASK_TYPES = {
	TaskType.WRITING: "Narrative and creative text generation.",
	TaskType.QUESTIONS: "Answering general inquiries.",
	TaskType.MATH: "Calculations and data interpretation.",
	TaskType.ROLEPLAY: "Simulated dialogues or scenarios.",
	TaskType.ANALYSIS: "Summarization, sentiment, or entity analysis.",
	TaskType.CREATIVITY: "Idea generation and design concepts.",
	TaskType.CODING: "Code assistance and generation.",
	TaskType.EDUCATION: "Teaching, explanations, and learning materials.",
	TaskType.RESEARCH: "Gathering and compiling information.",
	TaskType.TRANSLATION: "Text translation between languages.",
}


# Model decision logic
def get_model(difficulty: str, task: str) -> str:
	decision_tree = {
		TaskType.CODING: lambda d: "qwen2.5" if int(d) >= 4 else "deepseek-coder",
		TaskType.WRITING: lambda d: "deepseek-r1" if int(d) >= 3 else "gemma3",
		TaskType.CREATIVITY: lambda d: "deepseek-r1" if int(d) >= 3 else "gemma3",
		TaskType.MATH: lambda d: "gpt-4o" if int(d) >= 3 else "gemma3",
	}

	if task in decision_tree:
		return decision_tree[task](difficulty)
	elif difficulty in [DifficultyLevel.FOUR, DifficultyLevel.FIVE]:
		return "o4-mini"
	elif difficulty == DifficultyLevel.THREE:
		return "qwq"
	return "gemma3"


# Prompt template
system_prompt_template = """
You are an expert AI query classifier. Your task is to determine the difficulty level (1-5) and the most appropriate task type for user queries.

Difficulty Levels:
{{difficulty_details}}

Task Types:
{{task_types}}

Follow these rules carefully:
- Analyze the user query thoroughly to understand its complexity and intent.
- Based on your analysis and the provided Difficulty Levels and Task Types, classify the query into one Difficulty level (a single number from 1 to 5) and one Task Type (from the predefined categories).
- Ensure your output strictly adheres to the specified JSON format.

Output JSON Format:
{{{{"difficulty": "string", "task_type": "string"}}}}

Examples:
User query: Hello, how are you?
Output: {{{{"difficulty": "1", "task_type": "Questions"}}}}

User query: What's the capital of France?
Output: {{{{"difficulty": "2", "task_type": "Questions"}}}}

User query: Can you write a short story about a robot and a cat?
Output: {{{{"difficulty": "4", "task_type": "Writing"}}}}

User query: please write a code in python to calculate the sum of the first 10 Fibonacci numbers.
Output: {{{{"difficulty": "3", "task_type": "Coding"}}}}

User query: Write a Python function to add two numbers.
Output: {{{{"difficulty": "3", "task_type": "Coding"}}}}

User query: Act as a product manager and help me draft a user story for the login flow on my web application.
Output: {{{{"difficulty": "4", "task_type": "Roleplay"}}}}

User query: Explain the core concepts of quantum computing in simple terms.
Output: {{{{"difficulty": "4", "task_type": "Education"}}}}

User query: Compare and contrast the FastAPI and Flask frameworks for Python web development, highlighting their strengths and weaknesses.
Output: {{{{"difficulty": "5", "task_type": "Research"}}}}

User query: Please provide a Python code implementation for a user signup flow using FastAPI, including input validation.
Output: {{{{"difficulty": "5", "task_type": "Coding"}}}}

User query: Summarize the main arguments for and against universal basic income.
Output: {{{{"difficulty": "3", "task_type": "Analysis"}}}}

User query: Generate three distinct creative concepts for a new eco-friendly product aimed at reducing plastic waste in households.
Output: {{{{"difficulty": "4", "task_type": "Creativity"}}}}

User query: Translate the following sentence into Spanish: "The quick brown fox jumps over the lazy dog."
Output: {{{{"difficulty": "2", "task_type": "Translation"}}}}

User query: {{query}}
Output:
"""

# Fill static parts of the prompt
filled_prompt = system_prompt_template.format(
	difficulty_details=json.dumps(
		{k.value: v for k, v in DIFFICULTY_DETAILS.items()}, indent=2
	),
	task_types=json.dumps({k.value: v for k, v in TASK_TYPES.items()}, indent=2),
)

# LangChain components
router_model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=100)
output_parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template(filled_prompt)

router_chain = (
	{
		"query": lambda x: x,  # passthrough user input as 'query'
		"difficulty_details": lambda _: json.dumps(
			{k.value: v for k, v in DIFFICULTY_DETAILS.items()}, indent=2
		),
		"task_types": lambda _: json.dumps(
			{k.value: v for k, v in TASK_TYPES.items()}, indent=2
		),
	}
	| prompt
	| router_model
	| output_parser
)

# Main loop
while True:
	print("\n🔍 Enter your query below (or type 'exit' to quit):")
	user_input = input("> ").strip()

	if user_input.lower() == "exit":
		print("👋 Exiting. Have a great day!")
		break
	if not user_input:
		print("⚠️ Please enter a valid query.")
		continue

	try:
		response = router_chain.invoke({"query": user_input})
		difficulty = response.get("difficulty")
		task = response.get("task_type")

		print("\n🧠 Classification Result:")
		print(
			f" - Difficulty Level: {difficulty} ({DIFFICULTY_DETAILS.get(DifficultyLevel(difficulty), 'Unknown')})"
		)
		print(f" - Task Type: {task} ({TASK_TYPES.get(TaskType(task), 'Unknown')})")

		model_name = get_model(difficulty, task)
		print(f"\n🎯 Suggested Model for Your Query: {model_name}")

	except Exception as e:
		print(f"❌ An error occurred: {e}")
		print("⚙️ Falling back to default model: gemma3")

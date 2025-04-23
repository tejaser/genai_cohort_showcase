# LLM Router

## Pre requisite
- open ai api key
- langsmith/langchain key
- langsmith project name

## How to run
- install requirements from `requirements.txt` file
- run below command once env is created and api key are added to the `.env` file
```bash
python llm_router.py
```
## Functionality
- I have created a list of difficulties and task type
- llm will reference both list to classify the user query with a difficulty value between 1 to 5 and type of task like question-answer, coding, creativity, researtch etc.
- i then wrote a logic to pick model based on the combination of difficulty level and task type based on description written for model in their respective website.

## Next steps
- inplace of manual model selection it can be replaced by a random forest algorithm or bert classifier

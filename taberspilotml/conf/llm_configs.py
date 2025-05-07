from dataclasses import dataclass
from dataclasses_json import dataclass_json
import langchain_text_splitters as t_splitters
from dotenv import load_dotenv


# @dataclass(frozen=True)
class LLMConfigs:
    load_dotenv()  # Load environment variables from .env file
    embeddings_deployment: str = "text-embedding-ada-002"
    llm_deployment: str = "langchain_model"
    llm_type: str = 'azure_chat_openai'
    embeddings_type: str = 'azure_openai'
    openai_api_version: str = "2023-07-01-preview"
    return_retriever: bool = True
    text_splitter: t_splitters = None
    use_dynamic_prompt_recognizer: bool = True


# @dataclass_json


# @dataclass_json
# @dataclass(frozen=True)
# TODO: ADD PROMPTS WRAPPER FROM FILE agents_prompts.py
class PromptConfigs:
    answer_prompt: str = None
    agent_template_standard: str = None


@dataclass_json
@dataclass(frozen=True)
class Cfg:
    llm_configs: LLMConfigs = LLMConfigs()
    # database_configs: DatabaseConfigs = DatabaseConfigs()
    # flask_app_configs: FlaskAppConfigs = FlaskAppConfigs()
    # prompt_configs: PromptConfigs = PromptConfigs()
    # wiki_configs: WikiConfigs = WikiConfigs()

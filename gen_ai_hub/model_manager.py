import os
import ssl
import httpx
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from pydantic import SecretStr
from truststore import SSLContext

from gen_ai_hub.config import (
    GenAiHubEndpoint,
    ApiVersion,
    LanguageModel,
    EmbeddingModel,
)


# Factory Class
class ModelFactory:
    """
    Factory class to create GenAI Hub models.
    """

    @staticmethod
    def _get_http_client():
        return httpx.Client(verify=SSLContext(ssl.PROTOCOL_TLS_CLIENT))

    @staticmethod
    def _get_http_async_client():
        return httpx.AsyncClient(verify=SSLContext(ssl.PROTOCOL_TLS_CLIENT))

    @staticmethod
    def create_chat_model(
        api_key: SecretStr,
        azure_endpoint: GenAiHubEndpoint = GenAiHubEndpoint.PRD,
        api_version: ApiVersion = ApiVersion.V2024_06_01,
        deployment: LanguageModel = LanguageModel.GPT_4O,
        model: LanguageModel = LanguageModel.GPT_4O,
    ) -> AzureChatOpenAI:
        """
        Factory function to create a langchain chat model.
        :param api_key:
        :param azure_endpoint:
        :param api_version:
        :param deployment:
        :param model:
        :return:
        """
        return AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint.value,
            api_version=api_version.value,
            azure_deployment=deployment.value,
            model=model.value,
            http_client=ModelFactory._get_http_client(),
            http_async_client=ModelFactory._get_http_async_client(),
        )  # .with_structured_output(RequirementsList) # only possible with API versions of 2024-08-01 and later

    @staticmethod
    def create_embedding_model(
        api_key: SecretStr,
        azure_endpoint: GenAiHubEndpoint = GenAiHubEndpoint.PRD,
        api_version: ApiVersion = ApiVersion.V2024_06_01,
        deployment: EmbeddingModel = EmbeddingModel.TEXT_EMBEDDING_ADA_002,
    ) -> AzureOpenAIEmbeddings:
        """
        Factory function to create a langchain embedding model.
        :param api_key:
        :param azure_endpoint:
        :param api_version:
        :param deployment:
        :return:
        """
        return AzureOpenAIEmbeddings(
            api_key=api_key,
            azure_endpoint=azure_endpoint.value,
            api_version=api_version.value,
            azure_deployment=deployment.value,
            http_client=ModelFactory._get_http_client(),
            http_async_client=ModelFactory._get_http_async_client(),
        )

    @staticmethod
    def create_genai_hub_client(
        api_key: str,
        endpoint: GenAiHubEndpoint = GenAiHubEndpoint.PRD,
        api_version: ApiVersion = ApiVersion.V2024_06_01,
    ) -> AzureOpenAI:
        """
        Factory function to create an Azure OpenAI client.
        :param api_key:
        :param endpoint:
        :param api_version:
        :return:
        """
        openai_endpoint = endpoint.value + "/openai"
        return AzureOpenAI(
            base_url=openai_endpoint,
            api_key=api_key,
            api_version=api_version.value,
            http_client=ModelFactory._get_http_client(),
        )


# Usage Example
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    API_KEY = SecretStr(os.environ["API_KEY"])
    # Create instances using the factory
    chat_model = ModelFactory.create_chat_model(API_KEY)
    embedding_model = ModelFactory.create_embedding_model(API_KEY)
    genai_hub_client = ModelFactory.create_genai_hub_client(API_KEY)

    print("Chat Model, Embedding Model, and GenAI Hub Client created successfully.")

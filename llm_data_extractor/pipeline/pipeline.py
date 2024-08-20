from dotenv import load_dotenv
import pydantic
from pydantic.fields import FieldInfo
from langchain_community.document_loaders import PyPDFLoader
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def run_pipeline(
    text: str,
    fields_to_extract: dict[str, str]
) -> pydantic.BaseModel:
    load_dotenv()

    pydantic_fields_to_extract = {
        name: (str, FieldInfo(description=description))
        for name, description in fields_to_extract.items()
    }
    PydanticModel = pydantic.create_model(
        "PydanticModel",
       **pydantic_fields_to_extract
    )
    parser = PydanticOutputParser(pydantic_object=PydanticModel)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    chain = prompt | model | parser

    return chain.invoke(text)

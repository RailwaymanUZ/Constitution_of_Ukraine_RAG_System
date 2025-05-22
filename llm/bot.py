from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from loguru import logger
from pprint import pprint

import config
from .standart_prompt import prompt
from retriver import Retriever



class LLM:
    def __init__(self):
        self.__model = ChatOpenAI(model=config.MODEL_LLM, api_key=config.OPENAI_API_KEY)
        self.__prompt_template = PromptTemplate.from_template(prompt)
        self.__retriever = Retriever()
        self.result_rag = []


    def parse_doc(self, documents: list) -> str:
        """
        Method to parse answer retriever. Result to show save to self.result_rag.
        :param documents: list() with (Documents, score_rerank, score cosine distance) answer retriever
        :return: str() - part of prompt to replaced
        """
        result_to_prompt = ''
        for doc in documents:
            self.result_rag.append(
                {"documents_val": doc[0].page_content,
                 "score_rerank": float(doc[1]),
                 "score_cosine_distance": float(doc[-1])
                 }
            )
            result_to_prompt += f"{doc[0].page_content}\n\n"
        return result_to_prompt


    def _make_prompt(self, user_request: str) -> str:
        """
        Method to make prompt. (string request to LLM)
        :param user_request: str() user request
        :return: str() ready prompt to calling LLM
        """
        try:
            documents = self.__retriever.return_documents(user_request)
        except Exception as e:
            logger.error(f"Error with work retriever {e}")
            raise e

        rag_as_string = self.parse_doc(documents)
        request_text = self.__prompt_template.format(rag_documents=rag_as_string, user_request=user_request)
        return request_text

    def make_answer_dict(self, response_llm: BaseMessage) -> dict:
        answer = {
            "answer_llm": response_llm.content,
            "rag_documents": self.result_rag
        }
        return answer

    def request_to_llm(self, user_request: str) -> dict:
         """
         Method to realization request to LLM based on user request
         :param user_request: str() user request
         :return: dict() with key ("answer_llm": str,  "rag_documents" : list() (Documents, score rerank, score cosine)
         """
         request_text = self._make_prompt(user_request)

         try:
             response = self.__model.invoke(request_text)
         except Exception as e:
             logger.error(f"Problem with LLM {e}")
             raise e

         response = self.make_answer_dict(response)
         self.result_rag = []
         return response

    def work(self) -> None:
        """
        Method to run all app in console.
        Result will be printing on console.
        :return:
        """
        while True:
            try:
                user_request = str(input("**Будь ласка введіть своє питання**: "))
                result = self.request_to_llm(user_request)
                pprint(result)
            except Exception as e:
                logger.error(str(e))
                print("Сервісна помилка - подивіться в логи чи зверніться до розробника")

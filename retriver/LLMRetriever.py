import os
from loguru import logger
from scipy.spatial.distance import cosine
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from .abstract_class import AbstractRetriever
import config

class Retriever(AbstractRetriever):
    def __init__(self):
        super().__init__()
        self.__dirname = os.path.dirname(__file__)
        self.__fais_db = os.path.join(self.__dirname, 'faiss_index')
        self.__embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
        self.__db = FAISS.load_local(self.__fais_db, self.__embeddings, allow_dangerous_deserialization=True)
        self.__RETURN_DOCUMENTS_IN_RAG = config.RETURN_DOCUMENT_IN_RAG
        self.__SEARCH_DOCUMENTS_IN_RAG = config.SEARCH_DOCUMENTS_IN_RAG
        self.__CROSS_ENCODER_MODEL = config.CROSS_ENCODER_MODEL
        self.__RETURN_FINAL_DOCUMENTS = config.RETURN_FINAL_DOCUMENT
        self.__PARAM_MMR = config.PARAM_MMR
        self.__model_cross_encoder = HuggingFaceCrossEncoder(model_name=self.__CROSS_ENCODER_MODEL)
        self.__reranker_cross_encoder = CrossEncoderReranker(
            model=self.__model_cross_encoder,
            top_n=config.RETURN_FINAL_DOCUMENT
        )
        self.__retriever = self.__db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.__RETURN_DOCUMENTS_IN_RAG,
                "fetch_k": self.__SEARCH_DOCUMENTS_IN_RAG,
                "lambda_mult": self.__PARAM_MMR
            }
        )

    def return_retriever(self):
        return self.__retriever

    def cosine_distance(self, user_query: str, result_sort: list) -> list:
        """
        Method ro return score with cosine distance.
        This method was appended ro show cosine distance. In prod will this method removed
        :param user_query: str() user request
        :param result_sort: list() result answer retriever
        :return: list() with cosine distance score
        """
        result_sort = [element.page_content for element in result_sort]
        request_embed = self.__embeddings.embed_documents([user_query])[0]
        cosine_score = []
        for element in result_sort:
            cosine_score.append(
                cosine(request_embed, self.__embeddings.embed_documents([element])[0])
            )
        return cosine_score


    def return_documents(self, user_query: str) -> list:
        """
        Method to search relevant document in FAISS bd and return documents in list
        :param user_query: str() user_query or may be user prompt to LLM
        :return: list() with (Document, score cross encoder, cosine distance score) elements
        """
        try:
            answer = self.__retriever.invoke(user_query)
            cosine_score = self.cosine_distance(user_query, answer) # Remove this method. It is used to obtain the cosine distance.
        except Exception as e:
            logger.error(f"Problem with FAISS retriever {e}")
            raise e

        try:
            scores_rerank = self.__reranker_cross_encoder.model.score([(user_query, doc.page_content) for doc in answer])
        except Exception as e:
            logger.error(f"Problem with CROSS-ENCODER model {e}")
            raise e

        docs_with_scores = list(zip(answer, scores_rerank, cosine_score))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        docs_with_scores = docs_with_scores[:self.__RETURN_FINAL_DOCUMENTS]
        return docs_with_scores

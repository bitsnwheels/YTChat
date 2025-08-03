import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
from langdetect import detect
import re

def simple_sent_tokenize(text):
    """Splits text into sentences."""
    return re.split(r'(?<=[.?!])\s+', text.strip())

def maybe_translate(transcript: str) -> str:
    """Detects language and translates to English if necessary."""
    try:
        lang = detect(transcript)
    except:
        lang = "en"

    if lang == "en":
        return transcript
    else:
        sentences = simple_sent_tokenize(transcript)
        translated_sentences = [GoogleTranslator(source=lang, target='en').translate(sent) for sent in sentences]
        return " ".join(translated_sentences)

def get_chatbot_response(video_id: str, question: str) -> str:
    """Main function to get chatbot response."""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en", "hi"])
        
        raw_transcript = fetched.to_raw_data()
        transcript_text = " ".join(entry["text"] for entry in raw_transcript)
        
        

        translated_transcript = maybe_translate(transcript_text)

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
        chunks = splitter.create_documents([translated_transcript])

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.5}
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        prompt = PromptTemplate(
            template="""
              You are a helpful assistant.
              Answer ONLY from the provided transcript context.
              If the context is insufficient, just say you don't know.

              Context: {context}
              Question: {question}
            """,
            input_variables=["context", "question"]
        )

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        chain = (
            RunnableParallel({'context': retriever | format_docs, 'question': RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)
        return answer

    except TranscriptsDisabled:
        return "Sorry, transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No English or Hindi transcript found for this video."
    except VideoUnavailable:
        return "This video is unavailable."
    except Exception as e:
        return f"An error occurred: {e}"
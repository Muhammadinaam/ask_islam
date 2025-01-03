import os
from os import path

from langchain_community.vectorstores import FAISS
import openai
import pandas as pd
import numpy as np
from langdetect import detect

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
openai.api_key = OPENAI_API_KEY

islamic_chatbot = None


def get_islamic_chatbot():
    global islamic_chatbot
    if islamic_chatbot is None:
        current_directory = path.dirname(__file__)
        islamic_books_path = path.join(
            current_directory,
            '../../../data/islamic_books.csv'
        )
        islamic_chatbot = IslamicChatbot(islamic_books_path)
    return islamic_chatbot


class IslamicChatbot:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.chat_history = []  # Initialize chat history
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.csv_file)
        if 'english_text' not in self.df.columns:
            self.df['english_text'] = None
        if 'embedding' not in self.df.columns:
            self.df['embedding'] = None

        texts, embeddings, metadatas = [], [], []

        for idx, row in self.df.iterrows():
            print(idx)
            arabic_text = row['arabic_text']

            # Generate or use existing English translation
            if pd.isna(row['english_text']):
                self.df.at[idx, 'english_text'] = self.translate_to_english(
                    arabic_text)

            english_text = self.df.at[idx, 'english_text']

            # Generate or use existing embedding
            if pd.isna(row['embedding']) and english_text:
                embedding = self.generate_embedding(english_text)
                self.df.at[idx, 'embedding'] = str(embedding)
            else:
                embedding = eval(row['embedding']) if not pd.isna(
                    row['embedding']) else None

            if english_text and embedding:
                texts.append(english_text)
                embeddings.append(embedding)
                metadatas.append({
                    "book": row['book'],
                    "chapter_number": row['chapter_number'],
                    "chapter_name": row['chapter_name'],
                    "verse_or_hadith_number": row['verse_or_hadith_number'],
                    "embedding": embedding
                })

            # Save progress to CSV after processing each row
            self.df.to_csv(self.csv_file, index=False)

        # Create FAISS vector store
        if texts and embeddings:
            text_embeddings = list(zip(texts, embeddings))
            self.vector_store = FAISS.from_embeddings(
                embedding=None,
                text_embeddings=text_embeddings,
                metadatas=metadatas
            )
        else:
            self.vector_store = None

    def translate_to_english(self, text):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "assistant",
                    "content": ("You are a helpful assistant "
                                "that translates text to English. "
                                "Response should only contain translation "
                                "and no extra text.")},
                {
                    "role": "user",
                    "content": f"Translate this text to English: {text}"
                }
            ],
        )
        return response.choices[0].message.content.strip()

    def generate_embedding(self, text):
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002",
            encoding_format="float"
        )
        return response.data[0].embedding

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def query(self, user_query):
        user_language = self.detect_language(user_query)

        # Generate embedding for the user query
        user_query_embedding = self.generate_embedding(user_query)

        # Perform similarity search in the vector store
        if self.vector_store:
            relevant_texts = self.find_relevant_texts(user_query_embedding)

            response = []

            sources = []
            combined_text = ''
            for result in relevant_texts:
                metadata = result.metadata
                book = metadata.get('book', 'Unknown')
                chapter_number = metadata.get('chapter_number', 'Unknown')
                chapter_name = metadata.get('chapter_name', 'Unknown')
                verse_or_hadith_number = metadata.get(
                    'verse_or_hadith_number', 'Unknown')
                text = result.page_content
                combined_text = f'{combined_text} | {text}'

                sources.append({
                    "book": book,
                    "chapter_number": chapter_number,
                    "chapter_name": chapter_name,
                    "verse_or_hadith_number": verse_or_hadith_number,
                })

            prompt = (
                f"Answer this question: \"{user_query}\" "
                f"in language of the question."
                f"considering only this text: {combined_text}. Answer:"
            )

            explanation = self.llm(prompt)
            response.append({
                "sources": sources,
                "answer": explanation
            })

            # Update chat history
            self.chat_history.append({
                "user_query": user_query,
                "response": explanation
            })

            return response
        else:
            return "Vector store is not initialized."

    def find_relevant_texts(self, user_query_embedding):
        # Perform similarity search with k=10
        results = self.vector_store.similarity_search_by_vector(
            user_query_embedding, k=10)

        # Normalize embeddings (if not already normalized)
        user_query_embedding = user_query_embedding / np.linalg.norm(
            user_query_embedding)

        similarity_threshold = 0.8
        filtered_results = [
            result for result in results
            if np.dot(
                result.metadata['embedding'],
                user_query_embedding
            ) > similarity_threshold
        ]

        # Retrieve the top 3 results from the filtered list
        strict_results = filtered_results[:3]
        return strict_results

    def llm(self, prompt):
        try:
            messages = [
                {
                    "role": "assistant",
                    "content": "You are a helpful assistant."
                }
            ]
            for history in self.chat_history:
                messages.append(
                    {"role": "user", "content": history["user_query"]})
                messages.append(
                    {"role": "assistant", "content": history["response"]})
            messages.append({"role": "user", "content": prompt})

            completion = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error with LLM: {e}")
            return "Error generating response."

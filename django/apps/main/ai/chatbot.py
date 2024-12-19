import pickle
from os import path

import pandas as pd
from langchain_community.llms import HuggingFacePipeline, OpenAI
from langchain_community.vectorstores import FAISS
from langdetect import detect
from sentence_transformers import SentenceTransformer
from transformers import pipeline

islamic_chatbot = None

# Dictionary for model configuration
MODEL_CONFIG = {
    "huggingface": {
        "model_name": "EleutherAI/gpt-neo-1.3B",
        "type": "huggingface",
        "config": {
            "max_length": 512,
            "temperature": 0.2
        }
    },
    "openai": {
        "model_name": "text-davinci-003",
        "type": "openai",
        "config": {}
    },
    "free_openai": {
        "model_name": "gpt-3.5-turbo",  # Example for a free OpenAI model
        "type": "openai",
        "config": {}
    }
}


def get_islamic_chatbot():
    global islamic_chatbot

    if islamic_chatbot is None:
        current_directory = path.dirname(__file__)
        islamic_books_path = path.join(
            current_directory,
            '../../../data/islamic_books.csv')
        islamic_chatbot = IslamicChatbot(islamic_books_path)
    return islamic_chatbot


class IslamicChatbot:
    def __init__(self, csv_file, model_key="huggingface"):
        """
        Initialize the chatbot with the provided model type and model name.
        :param model_key: The key for selecting a model configuration from the 
        MODEL_CONFIG dictionary.
        """
        self.csv_file = csv_file

        # Model key for selecting from MODEL_CONFIG dictionary
        self.model_key = model_key
        # Get the model configuration from the dictionary
        self.model_config = MODEL_CONFIG[model_key]
        # Model name to be used
        self.model_name = self.model_config["model_name"]
        # Model type (huggingface or openai)
        self.model_type = self.model_config["type"]
        self.embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2')  # Sentence Transformers
        self.load_data()

    def load_data(self):
        """
        Load data, create embeddings, and initialize the FAISS vector store.
        """
        # Load the CSV data into a dataframe
        self.df = pd.read_csv(self.csv_file)

        # Combine adjacent verses based on the context size
        texts = []
        metadatas = []
        embeddings = []
        for idx, row in self.df.iterrows():
            print("dfindex: " + str(idx))
            context_size = 5 if row["book"] == "Quran" else 0

            # Collect previous verses
            previous_verses = []
            for i in range(1, context_size + 1):
                if idx - i >= 0 and self.df.iloc[idx - i]['book'] == "Quran":
                    previous_verses.insert(
                        0, self.df.iloc[idx - i]['arabic_text'])

            # Collect next verses
            next_verses = []
            for i in range(1, context_size + 1):
                if (idx + i < len(self.df) and
                        self.df.iloc[idx + i]['book'] == "Quran"):
                    next_verses.append(self.df.iloc[idx + i]['arabic_text'])

            # Combine previous, current, and next verses
            current_verse = row['arabic_text']
            combined_text = " ".join(
                previous_verses + [current_verse] + next_verses)

            # Compute the embedding for the combined text
            embedding = self.embedding_model.encode(combined_text).tolist()

            # Append the text, metadata, and embedding
            texts.append(combined_text)
            embeddings.append(embedding)
            metadatas.append({
                "book": row['book'],
                "chapter_number": row['chapter_number'],
                "chapter_name": row['chapter_name'],
                "verse_or_hadith_number": row['verse_or_hadith_number'],
            })

        # Initialize FAISS vector store
        text_embeddings = list(zip(texts, embeddings))
        self.vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=self.embedding_model,
            metadatas=metadatas
        )

    def save_state(
        self,
        file_path="vector_store.pkl",
        faiss_path="faiss_index"
    ):
        """
        Save the FAISS vector store and metadata to files.
        """
        # Save FAISS index
        self.vector_store.save_local(faiss_path)

        # Save metadata and additional information
        with open(file_path, "wb") as f:
            pickle.dump({
                "metadata": self.df[[
                    'book',
                    'chapter_number',
                    'chapter_name',
                    'verse_or_hadith_number']].to_dict('records')
            }, f)
        print(f"State saved to {file_path} and FAISS index to {faiss_path}")

    def load_state(
        self,
        file_path="vector_store.pkl",
        faiss_path="faiss_index"
    ):
        """
        Load the FAISS vector store and metadata from files.
        """
        # Load FAISS index
        self.vector_store = FAISS.load_local(
            faiss_path, embeddings=self.embedding_model)

        # Load metadata
        with open(file_path, "rb") as f:
            state = pickle.load(f)
            self.df = pd.DataFrame(state["metadata"])
        print("State loaded from "
              f"{file_path} and FAISS index from {faiss_path}")

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception:
            return "en"  # Default to English if detection fails

    def query(self, user_query):
        """
        Process user queries with dynamic context retrieval.
        :param user_query: The user's question.
        """
        # Detect user's language
        user_language = self.detect_language(user_query)

        # Convert user query to embeddings
        user_query_embedding = self.embedding_model.encode(user_query).tolist()

        # Use FAISS to retrieve top 3 relevant entries with query embedding
        relevant_texts = self.vector_store.similarity_search_by_vector(
            user_query_embedding, k=3
        )

        # Select LLM based on model type
        if self.model_type == "huggingface":
            hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                **self.model_config["config"]
            )
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
        elif self.model_type == "openai" or self.model_type == "free_openai":
            llm = OpenAI(model=self.model_name)

        # Prepare response with explanations
        response = []
        for result in relevant_texts:
            # Extract metadata
            metadata = result.metadata
            book = metadata.get('book', 'Unknown')
            chapter_number = metadata.get('chapter_number', 'Unknown')
            chapter_name = metadata.get('chapter_name', 'Unknown')
            verse_or_hadith_number = metadata.get(
                'verse_or_hadith_number', 'Unknown')

            # Retrieve the actual text
            combined_text = result.page_content

            # Create a contextual prompt for the LLM
            prompt = (
                "The following text is from an Islamic book. "
                f"Explain it in {user_language} "
                "considering the "
                f"context provided:\n\n{combined_text}\n\nExplanation:"
            )

            # Generate explanation using selected LLM
            explanation = llm(prompt)

            # Append response
            response.append({
                "book": book,
                "chapter_number": chapter_number,
                "chapter_name": chapter_name,
                "verse_or_hadith_number": verse_or_hadith_number,
                "answer": explanation
            })

        return response

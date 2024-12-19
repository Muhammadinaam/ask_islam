import pandas as pd
import os


def prepare_data():
    current_directory = os.path.dirname(__file__)
    quran_dataset_path = os.path.join(
        current_directory, 'The Quran Dataset.csv')
    df = pd.read_csv(quran_dataset_path)

    rows = []
    for index, df_row in df.iterrows():
        rows.append({
            "book": "Quran",
            "chapter_number": df_row["surah_no"],
            "chapter_name": (
                df_row["surah_name_ar"] + " - " + df_row["surah_name_roman"]),
            "verse_or_hadith_number": df_row["ayah_no_surah"],
            "arabic_text": df_row["ayah_ar"]
        })

    arranged_quran_df = pd.DataFrame(rows)
    arranged_quran_df.to_csv(os.path.join(
        current_directory, 'islamic_books.csv'))


prepare_data()

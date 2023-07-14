import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    # Read input from the website form
    ID = request.form['id']

    # Provide the file names, input ID, and output file name
    data_file = "data.csv"
    key_file = "key.csv"
    output_file = "output.txt"

    # Call the compare_csv_files function
    result, name = compare_csv_files(data_file, key_file, ID, output_file)

    if result.startswith("Output written"):
        with open(output_file, 'r') as file:
            search_query = file.read()
        # Load the CSV file
        data = pd.read_csv('Coursera.csv')

        # Preprocess the data
        course_names = data['Course Name'].fillna('')
        course_descriptions = data['Course Description'].fillna('')
        skills = data['Skills'].fillna('')
        course_urls = data['Course URL'].fillna('')

        # Combine the columns into a single text
        text_data = course_names + ' ' + course_descriptions + ' ' + skills + ' ' + course_urls

        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        # Fit and transform the text data
        tfidf_matrix = vectorizer.fit_transform(text_data)

        # Normalize the TF-IDF matrix
        normalized_tfidf = normalize(tfidf_matrix)

        # Function to perform search
        def search_engine(query, top_k=5):
            # Preprocess the query
            query = [query]

            # Transform the query using the trained vectorizer
            query_tfidf = vectorizer.transform(query)

            # Normalize the query TF-IDF vector
            normalized_query = normalize(query_tfidf)

            # Calculate cosine similarity between query and documents
            similarity_scores = cosine_similarity(normalized_query, normalized_tfidf)

            # Get top-k similar documents
            top_indices = np.argsort(similarity_scores.ravel())[::-1][:top_k]

            # Store the search results
            search_results = []

            # Append the top-k search results to the list
            for i, index in enumerate(top_indices):
                search_result = {
                    'course_name': data.loc[index, 'Course Name'],
                    'course_url': data.loc[index, 'Course URL'],
                    'university': data.loc[index, 'University'],
                    'course_rating': data.loc[index, 'Course Rating']
                }
                search_results.append(search_result)

            return search_results

        # Call the search_engine function
        search_results = search_engine(search_query)

        # Render the template with the search results and name
        return render_template('results.html', search_results=search_results, name=name)

    else:
        return result

def compare_csv_files(data_file, key_file, ID, output_file):
    # Read data.csv file and find technology, experience, and name for the given ID
    with open(data_file, 'r') as file:
        data_reader = csv.reader(file)
        next(data_reader)  # Skip the header row
        for row in data_reader:
            if row[2] == ID:
                technology = row[6]
                experience = row[3]
                name = row[0]
                break
        else:
            return "ID not found in data.csv", ""

    # Read key.csv file and find the keywords based on technology and experience
    with open(key_file, 'r') as file:
        key_reader = csv.reader(file)
        next(key_reader)  # Skip the header row
        for row in key_reader:
            if row[0] == technology and row[1] == experience:
                keywords = row[2]
                break
        else:
            return "No keywords found for the given technology and experience", ""

    # Write the output to a text file
    with open(output_file, 'w') as file:
        file.write(keywords)

    return "Output written to " + output_file, name

if __name__ == '__main__':
    app.run()

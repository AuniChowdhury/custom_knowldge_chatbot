# custom_knowldge_chatbot

This is a python-based chatbot application that can answer any questions about a specific document. It is built using OpenAI "text-davinci-003" model and the interface is generated using streamlit. 

## ##Prerequisites
1. Make sure to install all dependencies in requirements.txt
2. Generate OpenAI API_KEY
3. Add your API_KEY as an environment variable

## ##Description of contents

ingest.py -> This script is responsible to load the document that we want to feed into the GPT model, preprocess text by dividing them into chunks of allowable size, create embeddings and finally save then in 
a pickle file called "vectorstore.pkl". Run this script only once unless you want to add more documents later.

query_data.py -> As the name suggests, this script defines the query templates and also a function "get_chain" that takes the vector embeddings [we created earlier using ingest.py] as an argument and is responsible for chaining using langchain module.

main.py -> This is where everything comes together. This script loads the pickle file containing the embeddings. It creates and runs the chain obtained by using get_chain function and also generates a simple chatbot interface.

## ##How to run 
1. Create Virtual env and install dependencies in requirements.txt. 
2. Run ingest.py to create the "vectorstore.pkl" file.
3. Run the command "streamlit run main.py". This will launch the application in localhost.

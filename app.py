import streamlit as st
import requests
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient




embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key="HgZpQb9mGtu3BmkG9AtMQVhT9kRW9VHSqSXk2kqP")
llm = ChatGroq(temperature=0.5, groq_api_key="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", model_name="llama3-70b-8192")

qdrant_client = QdrantClient(
    url="https://f4bf0d03-f13c-43b6-87cc-32935393ab68.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="r8CbzHB7iEGgX5ISK885jtMkaHCLJYAceYvjx3V-OEtKHqn1xnzHvQ"
)


innovations = Qdrant(
        client=qdrant_client,
        collection_name="biomimicry-innovations",
        embeddings=embeddings
    )

strategies = Qdrant(
        client=qdrant_client,
        collection_name="biomimicry-strategies",
        embeddings=embeddings
    )
journals = Qdrant(
        client=qdrant_client,
        collection_name="journals",
        embeddings=embeddings
    )
patents = Qdrant(
        client=qdrant_client,
        collection_name="patents",
        embeddings=embeddings
    )



def qa_innovation(query, context):
    """
    This function sends a query and context to an LLM for biomimicry innovation insights.

    Args:
        query: The innovator's query about their innovation.
        context: The metadata content (string) from a similar biomimicry innovation.

    Returns:
        A string containing the LLM's response in markdown format.
    """
    prompt = f"""
    ## Biomimicry Innovation Insights

    **Query:** {query}

    **Context:**

    {context}

    **Task:** 

    Using the query provided just summarize the context. 

    If the context is not not related to the query then say you dont know the answer.

    In any case do not use any knowledge of your own.

    
    If the context matches the query and you are summarizing then I want you summarize in detail, include all the points in the context.

    **Answer:**
    """

    response = llm.invoke(prompt)
    return response.content



def qa_strategy(query, context):
    """
    This function sends a query and context to an LLM for biomimicry strategy insights.

    Args:
        query: The innovator's query about their biomimicry strategy.
        context: The metadata content (string) from a similar biomimicry strategy.

    Returns:
        A string containing the LLM's response in markdown format.
    """
    prompt = f"""
    ## Biomimicry Strategy Insights

    **Query:** {query}

    **Context:**

    {context}

    **Task:** 

    Using the query provided just summarize the context. 

    If the context is not not related to the query then say you dont know the answer.

    If the context matches the query and you are summarizing then I want you summarize in detail, include all the points in the context.

    In any case do not use any knowledge of your own.

    **Answer:**
    """

    response = llm.invoke(prompt)
    return response.content

def display_patents(found_docs):
    st.title("List of Patents")
    for idx, doc in enumerate(found_docs):
        st.markdown(f"## Patent {idx + 1}")
        st.write(f"**Title:** {doc.metadata['patent_title']}")
        st.write(f"**Type:** {doc.metadata['patent_type']}")
        st.write(f"**Date:** {doc.metadata['patent_date']}")
        st.write(f"**Abstract:** {doc.page_content}")
        st.write(f"**Number of Claims:** {doc.metadata['num_claims']}")
        st.write(f"**Assignee Name:** {doc.metadata['assignee_name']}")
        st.write(f"**Assignee Organization:** {doc.metadata['assignee_org']}")
        st.write(f"**Inventors:** {doc.metadata['inventors']}")
        st.write(f"**IPC:** {doc.metadata['ipc']}")
        st.write("---")

def display_journals(found_docs):
    st.title("List of Journals")
    for idx, doc in enumerate(found_docs):
        st.markdown(f"## Journal {idx + 1}")
        st.write(f"**Title:** {doc.page_content}")
        st.write(f"**Journal Name:** {doc.metadata['journal_name']}")
        st.write(f"**Publisher:** {doc.metadata['publisher']}")
        st.write(f"**Published Date:** {doc.metadata['published_date']}")
        st.write(f"**URL:** [{doc.page_content}]({doc.metadata['url']})")
        st.write(f"**Year:** {doc.metadata['year']}")
        st.write(f"**Updated:** {doc.metadata['updated']}")
        st.write(f"**Summary:** {doc.metadata['summary']}")
        st.write("---")

def search_query(query):
    # Base URL for the API
    base_url = "https://api.tavily.com/"

    # Endpoint for the search
    endpoint = "search"

    # Request payload
    payload = {
        "api_key": "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",
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 3,
        "include_domains": [],
        "exclude_domains": []
    }

    # Make the POST request
    response = requests.post(f"{base_url}{endpoint}", json=payload)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        data = response.json()
        
        # Print the answer
        st.write("Answer:", data["answer"])
        
        # Print top 3 titles with their URLs
        st.write("\nTop 3 Titles:")
        for result in data["results"][:3]:
            st.write(f"- {result['title']}: {result['url']}")



def main():

    # Streamlit app code
    st.title("Kreat.AI RAG test")

    #Sidebar Logo
    st.sidebar.image("logo.png")


    st.sidebar.title("Input Data")

    global query_string
    current_index = 0
    
    domain = st.sidebar.text_input("Domain")
    if domain:
        st.sidebar.markdown("✅ Domain entered")
    
    sub_domain = st.sidebar.text_input("Sub-Domain")
    if sub_domain:
        st.sidebar.markdown("✅ Sub-Domain entered")
    
    title = st.sidebar.text_input("Title")
    if title:
        st.sidebar.markdown("✅ Title entered")
    
        
    # Add button to discover innovations
    if st.sidebar.button("Discover Innovation",key = 'iv'):
        query = f"Domain: {domain}, Sub-Domain: {sub_domain}, Title: {title}"
        found_docs = innovations.similarity_search(query, k=3)
        
        st.markdown("## Innovation 1")
        context = found_docs[0].metadata['content']
        response = qa_innovation(query, context)
        # Display response in markdown format
        st.markdown(response)
        st.markdown("Reference: "+found_docs[0].metadata['Links'])

        prompt = f'''
                Give me some Biomimcry innovations regarding this query: {query}
                '''
        genai = llm.invoke(prompt)
        st.markdown("## Genai Innovations")
        st.write(genai.content)



    # Add button to discover strategies
    if st.sidebar.button("Discover Strategy",key='str'):
        query = f"Domain: {domain}, Sub-Domain: {sub_domain}, Title: {title}"
        found_docs = strategies.similarity_search(query, k=5)
        
        st.markdown("## Strategy 1")
        context = found_docs[0].metadata['content']
        response = qa_strategy(query, context)
        # Display response in markdown format
        st.markdown(response)
        st.write("Reference: "+found_docs[0].metadata['Reference'])
        prompt = f'''
                Give me some Biomimcry strategies regarding this query: {query}
                '''
        genai = llm.invoke(prompt)
        st.markdown("## Genai Strategies")
        st.write(genai.content)

        
    # Add button to discover patents
    if st.sidebar.button("Discover Patents",key='pt'):
        # Call qa function with the query and context
        # Replace query and context with appropriate values
        query = f"Domain: {domain}, Sub-Domain: {sub_domain}, Title: {title}"
        found_docs = patents.similarity_search(query, k=5)
        display_patents(found_docs)

    # Add button to discover journals
    if st.sidebar.button("Discover Journals",key='jr'):
        # Call qa function with the query and context
        # Replace query and context with appropriate values
        query = f"Domain: {domain}, Sub-Domain: {sub_domain}, Title: {title}"
        found_docs = journals.similarity_search(query, k=5)
        display_journals(found_docs)

    
    
    # Add button to discover search agents
    if st.sidebar.button("See latest",key='srch'):
        # Call qa function with the query and context
        # Replace query and context with appropriate values
        query = f"Give me latest trends for a problem having Domain: {domain}, Sub-Domain: {sub_domain} and Problem Title: {title}"
        search_query(query)


if __name__ == "__main__":
    main()

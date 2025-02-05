import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase

def init_database() -> SQLDatabase:
    """Initialize database connection using credentials from Streamlit Secrets."""
    try:
        db_uri = f"mysql+mysqlconnector://{st.secrets['database']['user']}:{st.secrets['database']['password']}@{st.secrets['database']['host']}:{st.secrets['database']['port']}/{st.secrets['database']['database']}"
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def getSQLChain(db):
    """Generate SQL query using LangChain's ChatGroq model."""
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=st.secrets["GROQ_API_KEY"])

    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Generate a natural language response based on SQL query results."""
    sql_chain = getSQLChain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! How can I assist you today?"),
    ]

st.set_page_config(page_title="Nouri Chatbot", page_icon="n_icon.jpeg", layout="wide")
st.title("Nouri AI CRM Assistant")

# Sidebar for database connection
with st.sidebar:
    st.subheader("Database Connection")
    st.write("This AI Chatbot assistant allows the customer to query their own data, better enhancing the user's ability to cultivate personal connections.")
    
    if st.button("Connect to Database"):
        with st.spinner("Connecting to database..."):
            db = init_database()
            if db:
                st.session_state.db = db
                st.success("Connected to database!")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI Assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("Ask a question about your data")

if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI Assistant"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
            
    st.session_state.chat_history.append(AIMessage(content=response))

# from langchain_openai import ChatOpenAI
# import streamlit as st
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.utilities import SQLDatabase

# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# def init_database(
#     user:str, password:str, host:str, port:str, database:str
# ) -> SQLDatabase:
#     db_uri = f"mysql+mysqlconnector://{st.secrets['database']['user']}:{st.secrets['database']['password']}@{st.secrets['database']['host']}:{st.secrets['database']['port']}/{st.secrets['database']['database']}"
#     return SQLDatabase.from_uri(db_uri)

# def getSQLChain(db):
#     template = """
#     You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
#     Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
#     <SCHEMA>{schema}</SCHEMA>
    
#     Conversation History: {chat_history}
    
#     Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
#     For example:
#     Question: which 3 artists have the most tracks?
#     SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
#     Question: Name 10 artists
#     SQL Query: SELECT Name FROM Artist LIMIT 10;
    
#     Your turn:
    
#     Question: {question}
#     SQL Query:
#     """
    
#     prompt = ChatPromptTemplate.from_template(template)

#     llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

#     def get_schema(_):
#         return db.get_table_info()
    
#     return(
#         RunnablePassthrough.assign(schema=get_schema)
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

# def get_response(user_query: str, db: SQLDatabase, chat_history: list):
#   sql_chain = getSQLChain(db)
  
#   template = """
#     You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
#     Based on the table schema below, question, sql query, and sql response, write a natural language response.
#     <SCHEMA>{schema}</SCHEMA>

#     Conversation History: {chat_history}
#     SQL Query: <SQL>{query}</SQL>
#     User question: {question}
#     SQL Response: {response}"""
  
#   prompt = ChatPromptTemplate.from_template(template)
  
#   llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
#   chain = (
#     RunnablePassthrough.assign(query=sql_chain).assign(
#       schema=lambda _: db.get_table_info(),
#       response=lambda vars: db.run(vars["query"]),
#     )
#     | prompt
#     | llm
#     | StrOutputParser()
#   )
  
#   return chain.invoke({
#     "question": user_query,
#     "chat_history": chat_history,
#   })
    


# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content="Hello! How can I assist you today?"),
#     ]



# st.set_page_config(page_title="Nouri Chatbot", page_icon="n_icon.jpeg", layout="wide")

# st.title("Nouri AI CRM Assistant")

# with st.sidebar:
#     st.subheader("Database Connection")
#     st.write("This AI Chatbot assistant allows the customer to query their own data, better enhancing the user's ability to cultivate personal connections.")
    
#     if st.button("Connect to Database"):
#         with st.spinner("Connecting to database..."):
#             db = init_database()
#             if db:
#                 st.session_state.db = db
#                 st.success("Connected to database!")

# # with st.sidebar:
# #     st.subheader("Database Connection")
# #     st.write("This AI Chatbot assistant allows the customer to query their own data, better enhancing the" 
# #     " user's ability to cultivate personal connections.")
    
# #     st.text_input("Host", value="localhost", key="Host")
# #     st.text_input("Port", value="3306", key="Port")
# #     st.text_input("User", value="root", key="User")
# #     st.text_input("Password", type="password", value="gray1380", key="Password")
# #     st.text_input("Database", value="sys", key="Database")
    
# #     if st.button("Connect"):
# #         with st.spinner("Connecting to database..."):
# #             db = init_database(
# #                 st.session_state["User"],
# #                 st.session_state["Password"],
# #                 st.session_state["Host"],
# #                 st.session_state["Port"],
# #                 st.session_state["Database"]
# #             )
# #             st.session_state.db = db
# #             st.success("Connected to database!")
    
# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI Assistant"):
#             st.markdown(message.content)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.markdown(message.content)

# user_query = st.chat_input("Ask a question about your data")

# if user_query is not None and user_query.strip() != "":
#     st.session_state.chat_history.append(HumanMessage(content=user_query))
    
#     with st.chat_message("Human"):
#         st.markdown(user_query)
    
#     with st.chat_message("AI Assistant"):
#         response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
#         st.markdown(response)
            
#     st.session_state.chat_history.append(AIMessage(content=response))
    


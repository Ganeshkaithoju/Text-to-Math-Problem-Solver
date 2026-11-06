import re
import statistics
import streamlit as st
import sympy as sp
import wikipedia
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain, LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool, initialize_agent

wikipedia.wikipedia.USER_AGENT = "GenAIMathApp/1.0 (ganeshkaithoju@gmail.com)"

load_dotenv()

st.title("Text to Math Problem solver using Groq LLama")
groq_api = st.sidebar.text_input("please enter your groq api key", type='password')

if not groq_api:
    st.info("Please enter your Groq API Key")
    st.stop()

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=SecretStr(groq_api),
    temperature=0,
    max_tokens=1000  
)

# NEW: safe wrapper so tool errors never crash the app 
UNDEFINED_PATTERN = re.compile(r'\b(inf|-inf|nan|zoo|oo)\b', re.IGNORECASE)

def safe_tool(func, check_undefined=False):
    def wrapper(query: str) -> str:
        try:
            out = str(func(query))
        except ZeroDivisionError:
            return ("ZeroDivisionError: division by zero is undefined. "
                    "Explain this to the user in simple words.")
        except Exception as e:
            return (f"Tool error ({type(e).__name__}): {e}. "
                    "Explain the problem to the user, or retry with a corrected input.")
        if check_undefined and UNDEFINED_PATTERN.search(out):
            out += ("\nNote: the result is undefined or infinite, "
                    "usually caused by division by zero. Explain this to the user.")
        return out
    return wrapper

# Tool 1: arithmetic via LLMMathChain 
math_chain = LLMMathChain.from_llm(llm=model)

calculator = Tool(
    name='Calculator',
    func=safe_tool(lambda q: math_chain.invoke({"question": q})["answer"], check_undefined=True),
    description=('Use for arithmetic and numeric math questions written in plain English '
                'or as an expression, e.g. "square root of 144", "20% of 50", "2^4".')
)

# Tool 2: algebra via sympy 
# Protecting the users from undefined results. As SymPy uses eval internally
SAFE_EXPR = re.compile(r'^[A-Za-z0-9\s.+\-*/^()=,]+$')
ALLOWED_NAMES = {"sqrt", "sin", "cos", "tan", "log", "exp", "pi", "E", "Abs"}

def algebra_tool_func(query: str) -> str:
    if not SAFE_EXPR.fullmatch(query) or len(query) > 200:
        raise ValueError("Only plain math expressions are allowed, e.g. x**2 - 4 = 0")
    
    # NEW: reject any multi-letter word that is not an approved math name
    for word in re.findall(r'[A-Za-z]+', query):
        if len(word) > 1 and word not in ALLOWED_NAMES:
            raise ValueError(f"'{word}' is not allowed. Use single-letter variables and basic math functions only.")

    query = query.replace("^", "**")
    if "=" in query:
        left, right = query.split("=", 1)
        return str(sp.solve(sp.Eq(sp.sympify(left), sp.sympify(right))))
    return str(sp.simplify(sp.sympify(query)))

algebra = Tool(
    name='Algebra Solver',
    func=safe_tool(algebra_tool_func, check_undefined=True),
    description=('Use to solve equations or simplify algebraic expressions with variables. '
                'Input is an equation such as "x**2 - 4 = 0" or an expression such as "(x+1)**2".')
)

# Tool 3: statistics via the statistics module 
def statistics_tool_func(query: str) -> str:
    op, nums = query.split(":", 1)
    data = [float(n) for n in nums.split(",")]
    ops = {"mean": statistics.mean, "median": statistics.median,
            "mode": statistics.mode, "stdev": statistics.stdev,
            "variance": statistics.variance}
    return str(ops[op.strip().lower()](data))

stats = Tool(
    name='Statistics',
    func=safe_tool(statistics_tool_func),
    description=('Use for mean, median, mode, stdev, variance of a list of numbers. '
                'Input format: "mean: 4, 8, 15, 16".')
)

# wikipedia tools
wikipedia_tool = Tool(
    name='Wikipedia',
    func=safe_tool(WikipediaAPIWrapper().run), # type: ignore
    description='Use for searching factual information.'
)

prompt_template = PromptTemplate(
    template=("You are an agent tasked with solving the user's logical and reasoning questions. Think step by step and give a clear final answer. "
            "Question: {question}\nAnswer:"),
    input_variables=["question"]
)
reasoning = Tool(
    name='Reasoning Tool',
    func=safe_tool(LLMChain(llm=model, prompt=prompt_template).run),
    description='Use for logical reasoning and word puzzles.'
)

# NEW: custom prefix tells the agent how to use history and how to handle errors
AGENT_PREFIX = (
    "You are a helpful math and general assistant. The input may include the earlier "
    "conversation. Use it to resolve references such as 'it', 'that result' or "
    "'my previous question', and answer questions about the conversation directly "
    "from it without using a tool. If a tool returns an error or an undefined result "
    "(for example division by zero), do not hide it: explain in plain language what "
    "went wrong and, where possible, how the user can fix the question. "
    "You have access to the following tools:"
)

assistant_agent = initialize_agent(
    tools=[calculator, algebra, stats, reasoning, wikipedia_tool],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    agent_kwargs={"prefix": AGENT_PREFIX}
)

# NEW: helpers for conversation history and error explanation 
def build_history(messages, limit=8):
    """Turn the last few chat messages into text for the agent."""
    recent = messages[-limit:]
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent
    )

def explain_error(question: str, error: Exception) -> str:
    """Last-resort: ask the LLM to explain an unexpected error to the user."""
    try:
        return model.invoke(
            f"A user asked: {question}\n"
            f"While answering, this error happened: {type(error).__name__}: {error}\n"
            "Explain in simple words what went wrong, and how the user can rephrase "
            "or fix the question. Keep it short."
        ).content # type: ignore
    except Exception:
        return f"Sorry, something went wrong ({type(error).__name__}: {error}). Please rephrase your question."

# Chat UI
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': 'Hello, I am your AI assistant. How can I help you today?'}
    ]

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

question = st.chat_input("Please enter your question here")

if question:
    # take history BEFORE adding the new question
    history = build_history(st.session_state.messages)

    st.session_state.messages.append({'role': 'user', 'content': question})
    st.chat_message('user').write(question)

    with st.spinner('Processing...'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        agent_input = f"Conversation so far:\n{history}\n\nCurrent question: {question}"
        try:
            response = assistant_agent.run(agent_input, callbacks=[st_cb])
        except Exception as e:
            response = explain_error(question, e)

    st.session_state.messages.append({'role': 'assistant', 'content': response})
    st.chat_message('assistant').write(response)

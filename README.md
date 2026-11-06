# Text-to-Math Problem Solver

A chat app where you type a question in plain English and an AI agent picks the right tool to answer it: a calculator, an algebra solver, a statistics tool, a reasoning chain, or Wikipedia. It remembers the recent conversation and explains errors, such as division by zero, in simple words instead of crashing.

Built with Streamlit, LangChain and Groq.

## Features

- Five tools, with the agent choosing one per question using the ReAct pattern (Thought → Action → Observation → Final Answer).
- Exact arithmetic and symbolic algebra, so numbers come from code instead of the model's guesses.
- Conversation memory: follow-ups like "multiply that by 2" work.
- Friendly error handling for undefined results, bad input and API failures.
- Live view of the agent's steps in the chat.
- API key entered in the sidebar and kept as a secret.

## How it works

```text
User question
     |
     v
Streamlit chat_input  →  build_history() adds the last 8 messages
     |
     v
ReAct agent (qwen/qwen3.8-27b on Groq)
     |── chooses → Calculator | Algebra Solver | Statistics | Reasoning | Wikipedia
     |                                   |
     |                       safe_tool() catches errors
     v                                   v
Final answer  ←──────────────  Observation (result or error text)
     |
     v
Shown in chat and saved in st.session_state
```

## Tech stack


| Library                         | Role in the app                                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `streamlit`                     | Chat UI, sidebar key box, spinner, and `session_state` to keep messages across reruns                   |
| `langchain-groq` (`ChatGroq`)   | Connects to Groq's hosted model that picks tools and writes answers                                     |
| `langchain-classic`             | `LLMMathChain`, `LLMChain`, `Tool`, `initialize_agent` and the `ZERO_SHOT_REACT_DESCRIPTION` agent type |
| `langchain-community`           | `WikipediaAPIWrapper` and `StreamlitCallbackHandler` (shows agent steps live)                           |
| `langchain-core`                | `PromptTemplate` for the reasoning prompt                                                               |
| `sympy`                         | Solving equations and simplifying expressions                                                           |
| `numexpr`                       | Used internally by `LLMMathChain` to evaluate expressions. Install it, though the code never imports it |
| `statistics` (standard library) | `mean`, `median`, `mode`, `stdev`, `variance`                                                           |
| `re` (standard library)         | Detects `inf`, `nan`, `zoo` and `oo` in tool output to flag undefined results                           |
| `wikipedia`                     | Imported to set a custom `USER_AGENT`, because Wikimedia blocks anonymous clients                       |
| `python-dotenv`                 | Loads environment variables from a `.env` file                                                          |
| `pydantic` (`SecretStr`)        | Keeps the API key out of logs and tracebacks                                                            |

## Getting started

Requires Python 3.10+ and a free [Groq API key](https://console.groq.com).

```bash
pip install streamlit langchain langchain-classic langchain-community langchain-groq \
            sympy numexpr wikipedia python-dotenv pydantic
streamlit run maths.py
```

Enter your Groq API key in the sidebar. Alternatively, store it in a `.env` file as `GROQ_API_KEY=...`. Never commit `.env`.

## Tools

The agent reads each tool's **description** to decide when to use it, so the descriptions are part of the app's logic.


| Tool           | Powered by                    | Used for                                         | Input                                            |
| ---------------- | ------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| Calculator     | `LLMMathChain` + numexpr      | Arithmetic, percentages, powers, roots           | Full question text, such as "square root of 144" |
| Algebra Solver | `sympy`                       | Equations and simplification                     | `x**2 - 4 = 0` or `(x+1)**2 - x**2`              |
| Statistics     | `statistics`                  | Mean, median, mode, standard deviation, variance | `operation: n1, n2, n3`                          |
| Reasoning Tool | `LLMChain` + `PromptTemplate` | Logic puzzles and word problems                  | The question, answered step by step              |
| Wikipedia      | `WikipediaAPIWrapper`         | Facts and definitions                            | A search topic                                   |

- **Calculator:** the LLM turns the question into an expression like `sqrt(144)`, and numexpr evaluates it exactly.
- **Algebra Solver:** if the input has `=`, it builds `sp.Eq(left, right)` and calls `sp.solve`. Otherwise it calls `sp.simplify`. For example, `x**2 - 4 = 0` returns `[-2, 2]`.
- **Statistics:** the text before the first `:` selects the operation, and the numbers after it are converted to floats, such as `mean: 4, 8, 15, 16`.
- **Wikipedia:** returns page summaries, which the agent rewrites as an answer.

## Memory and error handling

**Memory.** Streamlit reruns the script on every message, so the agent would forget by default. `st.session_state.messages` stores the chat and replays it on screen. `build_history()` joins the last 8 messages into text and adds it to the question as `Conversation so far: ... Current question: ...`. It is built before the new question is saved, so nothing is duplicated.

**Errors.** Every tool is wrapped in `safe_tool(func, check_undefined=False)`:

- It converts any result to a string, since agents work with text observations.
- A `ZeroDivisionError` becomes a clear message asking the agent to explain it simply.
- Any other exception becomes `Tool error (ExceptionName): message`, so the agent can explain it or retry with corrected input.
- With `check_undefined=True`, outputs containing `inf`, `nan`, `zoo` or `oo` get a note that the result is undefined or infinite.

The agent call also sits in a `try/except`. If something unexpected escapes, `explain_error()` asks the LLM to explain it in the context of the question. If that call fails too, the user sees a plain apology with the error type.

## Example questions

Answers are illustrative, since the model writes the final wording.


| Type       | Question                                                                    | Example answer                              |
| ------------ | ----------------------------------------------------------------------------- | --------------------------------------------- |
| Arithmetic | What is the square root of 144?                                             | 12                                          |
| Arithmetic | What is 20% of 50?                                                          | 10                                          |
| Algebra    | Solve x**2 - 4 = 0                                                          | x = -2 or x = 2                             |
| Algebra    | Simplify (x+1)**2 - x**2                                                    | 2*x + 1                                     |
| Statistics | Find the mean of 4, 8, 15, 16                                               | 10.75                                       |
| Statistics | Standard deviation of 2, 4, 4, 4, 5, 5, 7, 9                                | About 2.14 (sample)                         |
| Reasoning  | Ali is taller than Ben, and Ben is taller than Chitra. Who is the shortest? | Chitra                                      |
| Facts      | Who was Alan Turing?                                                        | A short biography from Wikipedia            |
| Follow-up  | What is 25 * 40? then Multiply that by 2                                    | 1000, then 2000                             |
| Error      | What is 10 divided by 0?                                                    | Explains that division by zero is undefined |
| Error      | Find the standard deviation of 5                                            | Asks for at least two values                |

## Project structure

```text
├── maths.py           # the Streamlit application
├── requirements.txt   # dependencies
├── .env.example       # GROQ_API_KEY placeholder (optional)
└── README.md
```

## Limits and tips

- **Model choice:** `LLMMathChain` and the ReAct agent rely on exact text formats and stop sequences. In my testing `qwen/qwen3.8-27b` passed the `LLMMathChain` check, while `openai/gpt-oss-20b` and `gpt-oss-120b` failed it. Re-test if you switch models.
- **`max_tokens`:** on Groq's on-demand tier a high value can trigger rate-limit (429) errors with this model, so lower it if you see them. Reasoning models that spend tokens on thinking may need a higher value elsewhere, or answers get cut off.
- **History length:** `build_history(limit=8)` keeps token use low. Raise `limit` for longer memory.
- **Wikipedia:** keep a descriptive `USER_AGENT` with a real contact email, or requests may be blocked.
- **Security:** the algebra tool parses text with `sympify`, which uses `eval` internally. Keep an input check in front of it before exposing the app publicly.
- **Adding a tool:** write a function that takes a string and returns a string, wrap it with `safe_tool`, give it a clear `description`, and add it to the `tools` list. Calculus with `sympy.diff` and matrices with `numpy` are good next steps.

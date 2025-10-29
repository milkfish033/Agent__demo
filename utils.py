import pandas as pd
from IPython.display import HTML, display
import re
import matplotlib.pyplot as plt
import openai

# ======================
# Data Loading Utilities
# ======================

def load_and_prepare_data(filepath):
    """
    Load a CSV into a pandas DataFrame and perform light cleaning.
    """
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip()           # clean extra spaces
    df = df.dropna(how="all")                     # remove empty rows
    df = df.dropna(axis=1, how="all")             # remove empty columns

    return df


def print_html(text_or_df, title="Output"):
    """
    Display either DataFrame or plain text/HTML nicely in Jupyter.
    """
    if isinstance(text_or_df, pd.DataFrame):
        html = f"<h3>{title}</h3>" + text_or_df.to_html(border=0)
        display(HTML(html))
    else:
        html = f"<h3>{title}</h3><pre>{text_or_df}</pre>"
        display(HTML(html))


# ======================
# LLM Wrapper
# ======================

def get_response(model: str, prompt: str):
    """
    Send prompt to an OpenAI-compatible LLM model.
    Replace this with your actual API call if needed.
    """
    client = openai.OpenAI()  # assumes env var OPENAI_API_KEY is set

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# ======================
# Code Extraction Utilities
# ======================

def extract_python_code(tagged_text: str) -> str:
    """
    Extract code wrapped in <execute_python>...</execute_python>.
    Returns only the code block.
    """
    match = re.search(r"<execute_python>(.*?)</execute_python>", tagged_text, re.DOTALL)
    if not match:
        raise ValueError("No <execute_python> code block found in the LLM response.")
    return match.group(1).strip()


def execute_code(code_str: str, df):
    """
    Executes generated python code safely in a controlled namespace.
    df is provided so generated code can use it directly.
    """
    namespace = {"df": df, "pd": pd, "plt": plt}
    exec(code_str, namespace)
    plt.close("all")  # ensure no charts auto-display

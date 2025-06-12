import time
from IPython.display import display, Image, Markdown
from google import genai
from google.genai import types
from google.api_core import retry
import numpy as np
import pandas as pd

def show_response(response):
    for p in response.candidates[0].content.parts:
        if p.text:
            display(Markdown(p.text))
        elif p.inline_data:
            display(Image(p.inline_data.data))
        else:
            print(p.to_json_dict())
        display(Markdown('----'))

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 502, 503})

@retry.Retry(predicate=is_retriable)
def generate_embedding(
    client: genai.Client, model: str, contents: str, task_type: str):
    response = client.models.embed_content(
        model=model,
        contents=contents,
        config=types.EmbedContentConfig(task_type=task_type)
        )
    return response

def embed_ethics(
        df: pd.DataFrame, col: str, embeddings_dfs: dict,
        failed_response: dict, embedding_client: genai.Client, model: str):
    t0 = time.time()
    t0_total = time.time()
    elasped_time = 0
    total_elasped_time = 0
    for task_type in embeddings_dfs:
        for i, row in df.iterrows():
            elasped_time = time.time() - t0
            total_elasped_time = time.time() - t0_total
            if i % 100 == 0:
                print(f'On task {task_type} and index {i} at {total_elasped_time:.2f} seconds.')
            if elasped_time >= 30:
                print('Sleeping for 90 seconds to reduce chance of hitting request per minute limit.')
                time.sleep(90)
                t0 = time.time()
            if i not in embeddings_dfs[task_type].index:
                response = generate_embedding(
                    embedding_client, model, row[col], task_type)
                if response.embeddings:
                    embeddings_dfs[task_type].loc[i] = response.embeddings[0].values
                else:
                    failed_response[(i, task_type)] = response.model_dump()

def chpt_verse_str_to_int(s):
    s = s.strip()
    if s:
        return int(s)
    # TODO: return pd.NA instead
    return np.nan

def truncate(t: str, limit: int=50) -> str:
  """Truncate labels to fit on the chart."""
  if len(t) > limit:
    return t[:limit-3] + '...'
  else:
    return t
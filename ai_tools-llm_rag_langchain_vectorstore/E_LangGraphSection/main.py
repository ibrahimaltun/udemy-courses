import sys
import os

# ---------------------  SET ENV   ---------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from A_IntroTopics.a_set_env_key import set_env

set_env()

# ---------------------  Imports ---------------------
from graph.graph import app

if __name__ == "__main__":
    print(app.invoke(input={"question": "What is prompt engineer?"}))

from node_constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from nodes import generate, grade_documents, web_search, retrieve
from chains.router import question_router, RouteQuery
from state import GraphState
from chains.halucination_grader import hallucination_grader
from chains.answer_grader import answer_grader

from langgraph.graph import END, StateGraph

# ---------------------  SET ENV   ---------------------
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from A_IntroTopics.a_set_env_key import set_env

set_env()


# ---------------------  LANGGRAPH WORKFLOW   ---------------------

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
